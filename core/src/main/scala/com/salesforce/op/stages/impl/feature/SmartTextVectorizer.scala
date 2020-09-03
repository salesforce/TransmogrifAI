/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.VectorizerUtils._
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.stages.SensitiveFeatureMode.Off
import com.salesforce.op.utils.stages.{NameDetectFun, NameDetectStats}
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.{Monoid, Semigroup, Tuple2Semigroup}
import org.apache.spark.ml.param._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, Encoders}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

/**
 * Convert a sequence of text features into a vector by detecting categoricals that are disguised as text.
 * A categorical will be represented as a vector consisting of occurrences of top K most common values of that feature
 * plus occurrences of non top k values and a null indicator (if enabled).
 * Non-categoricals will be converted into a vector using the hashing trick. In addition, a null indicator is created
 * for each non-categorical (if enabled).
 *
 * Detection and removal of names in the input columns can be enabled with the `sensitiveFeatureMode` param.
 *
 * @param uid uid for instance
 */
class SmartTextVectorizer[T <: Text](uid: String = UID[SmartTextVectorizer[T]])(implicit tti: TypeTag[T])
  extends SequenceEstimator[T, OPVector](operationName = "smartTxtVec", uid = uid)
    with PivotParams with CleanTextFun with SaveOthersParams
    with TrackNullsParam with MinSupportParam with TextTokenizerParams with TrackTextLenParam
    with HashingVectorizerParams with HashingFun with OneHotFun
    with MaxCardinalityParams with MinLengthStdDevParams
    with NameDetectFun[T] {
  private implicit val textStatsSeqEnc: Encoder[Array[TextStats]] = ExpressionEncoder[Array[TextStats]]()
  type StatsTuple = (TextStats, NameDetectStats)
  type StatsTupleArray = Array[StatsTuple]
  private implicit val statsTupleSeqEnc: Encoder[StatsTupleArray] = Encoders.kryo[StatsTupleArray]

  private def makeHashingParams() = HashingFunctionParams(
    hashWithIndex = $(hashWithIndex),
    prependFeatureName = $(prependFeatureName),
    numFeatures = $(numFeatures),
    numInputs = inN.length,
    maxNumOfFeatures = TransmogrifierDefaults.MaxNumOfFeatures,
    binaryFreq = $(binaryFreq),
    hashAlgorithm = getHashAlgorithm,
    hashSpaceStrategy = getHashSpaceStrategy
  )

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    require(!dataset.isEmpty, "Input dataset cannot be empty")

    val maxCard = $(maxCardinality)
    val minLenStdDev = $(minLengthStdDev)
    val shouldCleanText = $(cleanText)
    val shouldTokenizeForLengths = $(textLengthType) == TextLengthType.Tokens.entryName

    implicit val textStatsMonoid: Semigroup[TextStats] = TextStats.monoid(maxCard)
    implicit val nameDetectStatsMonoid: Semigroup[NameDetectStats] = NameDetectStats.monoid
    implicit val statsTupleMonoid: Semigroup[StatsTuple] = new Tuple2Semigroup[TextStats, NameDetectStats]()

    val (aggregatedStats, aggNameDetectStats): (Array[TextStats], Array[NameDetectStats]) =
      if (getSensitiveFeatureMode == Off) {
        (dataset.map(_.map(TextStats.computeTextStats(_, shouldCleanText, shouldTokenizeForLengths, maxCard)).toArray)
          .reduce(_ + _),
        Array.fill[NameDetectStats](inN.length)(NameDetectStats.empty))
    } else {
      val nameDetectMapFun = makeMapFunction(dataset.sparkSession)
      dataset.map(_.map(input => (TextStats.computeTextStats(input, shouldCleanText, shouldTokenizeForLengths, maxCard),
          nameDetectMapFun(input))
      ).toArray).reduce(_ + _).unzip
    }
    // val valueStats: Dataset[Array[TextStats]] = dataset.map(
    //   _.map(TextStats.computeTextStats(_, shouldCleanText, shouldTokenizeForLengths, maxCard)).toArray
    // )
    // val aggregatedStats: Array[TextStats] = valueStats.reduce(_ + _)

    val isNameArr: Array[Boolean] = aggNameDetectStats.map(computeTreatAsName)

    val topKValue = $(topK)
    val (vectorizationMethods, topValues) = aggregatedStats.zip(isNameArr).map { case (stats, isName) =>
      // Estimate the coverage of the top K values
      val totalCount = stats.valueCounts.values.sum // total count
      // Filter by minimum support
      val filteredStats = stats.valueCounts.filter { case (_, count) => count >= $(minSupport) }
      val sorted = filteredStats.toSeq.sortBy(- _._2)
      val sortedValues = sorted.map(_._2)
      // Cumulative Count
      val cumCount = sortedValues.headOption.map(_ => sortedValues.tail.scanLeft(sortedValues.head)(_ + _))
        .getOrElse(Seq.empty)
      val coverage = cumCount.lift(math.min(topKValue, cumCount.length) - 1).getOrElse(0L) * 1.0 / totalCount
      val vecMethod: TextVectorizationMethod = stats match {
          case _ if isName && shouldRemoveSensitive => TextVectorizationMethod.Ignore
          // If cardinality not respected, but coverage is, then pivot the feature
          // Extra checks need to be passed :
          //
          //  - Cardinality must be greater than maxCard (already mentioned above).
          //  - Cardinality must also be greater than topK.
          //  - Finally, the computed coverage of the topK with minimum support must be > 0.
          case _ if stats.valueCounts.size > maxCard && stats.valueCounts.size > topKValue && coverage >= $(coveragePct)
          => TextVectorizationMethod.Pivot
          case _ if stats.valueCounts.size <= maxCard => TextVectorizationMethod.Pivot
          case _ if stats.lengthStdDev < minLenStdDev => TextVectorizationMethod.Ignore
          case _ => TextVectorizationMethod.Hash
        }
      val topValues = filteredStats
        .toSeq.sortBy(v => -v._2 -> v._1)
        .take(topKValue).map(_._1)
      (vecMethod, topValues)
    }.unzip

    val smartTextParams = SmartTextVectorizerModelArgs(
      vectorizationMethods = vectorizationMethods,
      topValues = topValues,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = $(trackNulls),
      hashingParams = makeHashingParams()
    )

    val vecMetadata = makeVectorMetadata(smartTextParams, aggNameDetectStats)
    logInfo("TextStats for features used in SmartTextVectorizer:")
    inN.map(_.name).zip(aggregatedStats).zip(vectorizationMethods).foreach { case((name, stats), vecMethod) =>
      logInfo(s"Feature: $name")
      logInfo(s"LengthCounts: ${stats.lengthCounts}")
      logInfo(s"LengthMean: ${stats.lengthMean}")
      logInfo(s"LengthStdDev: ${stats.lengthStdDev}")
      logInfo(s"Vectorization method: $vecMethod")
    }
    setMetadata(vecMetadata.toMetadata)

    new SmartTextVectorizerModel[T](args = smartTextParams, operationName = operationName, uid = uid)
      .setAutoDetectLanguage(getAutoDetectLanguage)
      .setAutoDetectThreshold(getAutoDetectThreshold)
      .setDefaultLanguage(getDefaultLanguage)
      .setMinTokenLength(getMinTokenLength)
      .setToLowercase(getToLowercase)
      .setTrackTextLen($(trackTextLen))
      .setStripHtml(getStripHtml)
  }

  private def makeVectorMetadata(
    smartTextParams: SmartTextVectorizerModelArgs,
    aggNameDetectStats: Array[NameDetectStats]
  ): OpVectorMetadata = {
    require(inN.length == smartTextParams.vectorizationMethods.length)

    val groups = inN.toArray.zip(smartTextParams.vectorizationMethods).groupBy(_._2)
    val textToPivot = groups.getOrElse(TextVectorizationMethod.Pivot, Array.empty).map(_._1)
    val textToIgnore = groups.getOrElse(TextVectorizationMethod.Ignore, Array.empty).map(_._1)
    val textToHash = groups.getOrElse(TextVectorizationMethod.Hash, Array.empty).map(_._1)
    val allTextFeatures = textToHash ++ textToIgnore

    // build metadata describing output
    val shouldTrackNulls = $(trackNulls)
    val shouldTrackLen = $(trackTextLen)
    val unseen = Option($(unseenName))

    val categoricalColumns = if (textToPivot.nonEmpty) {
      makeVectorColumnMetadata(shouldTrackNulls, unseen, smartTextParams.categoricalTopValues, textToPivot)
    } else Array.empty[OpVectorColumnMetadata]

    val textColumns = if (textToHash.nonEmpty) {
      makeVectorColumnMetadata(textToHash, makeHashingParams())
    } else Array.empty[OpVectorColumnMetadata]

    val nullAndLenColumns = if (allTextFeatures.nonEmpty) {
      if (shouldTrackLen) {
        allTextFeatures.map(_.toColumnMetaData(descriptorValue = OpVectorColumnMetadata.TextLenString)) ++
        allTextFeatures.map(_.toColumnMetaData(isNull = true))
      }
      else {
        allTextFeatures.map(_.toColumnMetaData(isNull = true))
      }
    } else Array.empty[OpVectorColumnMetadata]
    val columns = categoricalColumns ++ textColumns ++ nullAndLenColumns

    val sensitive = createSensitiveFeatureInformation(aggNameDetectStats, inN.map(_.name))

    OpVectorMetadata(getOutputFeatureName, columns, Transmogrifier.inputFeaturesToHistory(inN, stageName), sensitive)
  }
}

object SmartTextVectorizer {
  val MaxCardinality: Int = 1000
  val MinTextLengthStdDev: Double = 0
  val LengthType: TextLengthType = TextLengthType.FullEntry
  val CoveragePct: Double = 0.90

  private[op] def partition[T: ClassTag](input: Array[T], condition: Array[Boolean]): (Array[T], Array[T]) = {
    val all = input.zip(condition)
    (all.collect { case (item, true) => item }, all.collect { case (item, false) => item })
  }
}

/**
 * Summary statistics of a text feature
 *
 * @param valueCounts  counts of feature values
 * @param lengthCounts counts of token lengths
 */
private[op] case class TextStats(valueCounts: Map[String, Long], lengthCounts: Map[Int, Long]) extends JsonLike {
  val lengthSize = lengthCounts.values.sum
  val lengthMean: Double = lengthCounts.foldLeft(0.0)((acc, el) => acc + el._1 * el._2) / lengthSize
  val lengthVariance: Double = lengthCounts.foldLeft(0.0)(
    (acc, el) => acc + el._2 * (el._1 - lengthMean) * (el._1 - lengthMean)
  ) / lengthSize
  val lengthStdDev: Double = math.sqrt(lengthVariance)
}

private[op] object TextStats extends CleanTextFun {
  /**
   * Helper function to add two maps subject to a max cardinality restriction on the number of unique values
   *
   * @param totalMap        Current accumulated map
   * @param mapToAdd        Additional map to add the to accumulated one
   * @param maxCardinality  Maximum number of unique keys to keep track of (stop counting once this is hit)
   * @tparam T              Type parameter for the keys
   * @return                Newly accumulated map subject to the key cardinality constraints
   */
  def additionHelper[T](totalMap: Map[T, Long], mapToAdd: Map[T, Long], maxCardinality: Int): Map[T, Long] = {
    if (totalMap.size > maxCardinality) totalMap
    else if (mapToAdd.size > maxCardinality) mapToAdd
    else totalMap + mapToAdd
  }

  def monoid(maxCardinality: Int): Monoid[TextStats] = new Monoid[TextStats] {
    override def plus(l: TextStats, r: TextStats): TextStats = {
      val newValueCounts = additionHelper(l.valueCounts, r.valueCounts, maxCardinality)
      val newLengthCounts = additionHelper(l.lengthCounts, r.lengthCounts, maxCardinality)
      TextStats(newValueCounts, newLengthCounts)
    }

    override def zero: TextStats = TextStats.empty
  }

  def empty: TextStats = TextStats(Map.empty, Map.empty)

  /**
   * Computes a TextStats instance from a Text entry
   *
   * @param text            Text value (eg. entry in a dataframe)
   * @param shouldCleanText Whether or not the text should be cleaned. Note that this only makes sense if tokenization
   *                        is not employed, since that will already do the cleaning steps (and more!)
   * @param shouldTokenize  Whether or not the text should be tokenized for length counts. If false, then the length
   *                        will just be the length of the entire entry
   * @param maxCardinality  Max cardinality to keep track of in maps (relevant for the text length distribution here)
   * @tparam T              Feature type that the text value is coming from
   * @return                TextStats instance with value and length counts filled out appropriately
   */
  private[op] def computeTextStats[T <: Text : TypeTag](
    text: T#Value,
    shouldCleanText: Boolean,
    shouldTokenize: Boolean,
    maxCardinality: Int
  ): TextStats = {
    text match {
      case Some(v) => textStatsFromString(v, shouldCleanText, shouldTokenize, maxCardinality)
      case None => TextStats(Map.empty[String, Long], Map.empty[Int, Long])
    }
  }

  /**
   * Computes a TextStats instance from a String entry
   *
   * @param textString      String to convert to TextStats
   * @param shouldCleanText Whether or not the text should be cleaned. Note that this only makes sense if tokenization
   *                        is not employed, since that will already do the cleaning steps (and more!)
   * @param shouldTokenize  Whether or not the text should be tokenized for length counts. If false, then the length
   *                        will just be the length of the entire entry
   * @param maxCardinality  Max cardinality to keep track of in maps (relevant for the text length distribution here)
   * @return                TextStats instance with value and length counts filled out appropriately
   */
  private[op] def textStatsFromString(
    textString: String,
    shouldCleanText: Boolean,
    shouldTokenize: Boolean,
    maxCardinality: Int
  ): TextStats = {
    // Go through each token in text and start appending it to a TextStats instance
    val lengthsMap = if (shouldTokenize) {
      TextTokenizer.tokenizeString(textString).tokens.value
        .foldLeft(Map.empty[Int, Long])(
          (acc, el) => TextStats.additionHelper(acc, Map(el.length -> 1L), maxCardinality)
        )
    } else Map(cleanTextFn(textString, shouldCleanText).length -> 1L)
    TextStats(Map(cleanTextFn(textString, shouldCleanText) -> 1L), lengthsMap)
  }
}

/**
 * Arguments for [[SmartTextVectorizerModel]]
 *
 * @param vectorizationMethods method to use for text vectorization (either pivot, hashing, or ignoring)
 * @param topValues            top values to each feature
 * @param shouldCleanText      should clean text value
 * @param shouldTrackNulls     should track nulls
 * @param hashingParams        hashing function params
 */
case class SmartTextVectorizerModelArgs
(
  vectorizationMethods: Array[TextVectorizationMethod],
  topValues: Array[Seq[String]],
  shouldCleanText: Boolean,
  shouldTrackNulls: Boolean,
  hashingParams: HashingFunctionParams
) extends JsonLike {
  def categoricalTopValues: Array[Seq[String]] = {
    topValues.zip(vectorizationMethods.map(_ == TextVectorizationMethod.Pivot)).collect { case (top, true) => top }
  }
}

final class SmartTextVectorizerModel[T <: Text] private[op]
(
  val args: SmartTextVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T]) extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
  with TextTokenizerParams with TrackTextLenParam with HashingFun with OneHotModelFun[Text] {

  override protected def convertToSet(in: Text): Set[String] = in.value.toSet

  def transformFn: Seq[Text] => OPVector = {
    val categoricalPivotFn: Seq[Text] => OPVector = pivotFn(
      topValues = args.categoricalTopValues,
      shouldCleanText = args.shouldCleanText,
      shouldTrackNulls = args.shouldTrackNulls
    )
    (row: Seq[Text]) => {
      val groups = row.toArray.zip(args.vectorizationMethods).groupBy(_._2)
      val textToPivot = groups.getOrElse(TextVectorizationMethod.Pivot, Array.empty).map(_._1)
      val textToIgnore = groups.getOrElse(TextVectorizationMethod.Ignore, Array.empty).map(_._1)
      val textToHash = groups.getOrElse(TextVectorizationMethod.Hash, Array.empty).map(_._1)

      val categoricalVector: OPVector = categoricalPivotFn(textToPivot)
      val textTokens: Seq[TextList] = textToHash.map(tokenize(_).tokens)
      val ignorableTextTokens: Seq[TextList] = textToIgnore.map(tokenize(_).tokens)
      val textVector: OPVector = hash[TextList](textTokens, getTextTransientFeatures, args.hashingParams)
      val textNullIndicatorsVector = if (args.shouldTrackNulls) {
        getNullIndicatorsVector(textTokens ++ ignorableTextTokens)
      } else OPVector.empty
      val textLenVector = if ($(trackTextLen)) getLenVector(textTokens ++ ignorableTextTokens) else OPVector.empty

      categoricalVector.combine(textVector, textLenVector, textNullIndicatorsVector)
    }
  }

  private def getTextTransientFeatures: Array[TransientFeature] =
    getTransientFeatures().zip(args.vectorizationMethods).collect {
      case (tf, method) if method != TextVectorizationMethod.Pivot => tf
    }

  private def getNullIndicatorsVector(textTokens: Seq[TextList]): OPVector = {
    val nullIndicators = textTokens.map { tokens =>
      val nullVal = if (tokens.isEmpty) 1.0 else 0.0
      Seq(0 -> nullVal)
    }
    val reindexed = reindex(nullIndicators)
    val vector = makeSparseVector(reindexed)
    vector.toOPVector
  }

  private def getLenVector(textTokens: Seq[TextList]): OPVector = {
    textTokens.map(f => f.value.map(_.length).sum.toDouble).toOPVector
  }
}

trait MaxCardinalityParams extends Params {
  final val maxCardinality = new IntParam(
    parent = this, name = "maxCardinality",
    doc = "max number of distinct values a categorical feature can have",
    isValid = ParamValidators.inRange(lowerBound = 1, upperBound = SmartTextVectorizer.MaxCardinality)
  )
  final def setMaxCardinality(v: Int): this.type = set(maxCardinality, v)
  final def getMaxCardinality: Int = $(maxCardinality)
  setDefault(maxCardinality -> SmartTextVectorizer.MaxCardinality)

  final val coveragePct = new DoubleParam(
    parent = this, name = "coveragePct",
    doc = "Threshold of percentage of the entries. If the topK entries make up for more than this pecentage," +
      " the feature is treated as a categorical",
    isValid = ParamValidators.inRange(lowerBound = 0, upperBound = 1, lowerInclusive = false, upperInclusive = true)
  )
  final def setCoveragePct(v: Double): this.type = set(coveragePct, v)
  final def getCoveragePct: Double = $(coveragePct)
  setDefault(coveragePct -> SmartTextVectorizer.CoveragePct)
}

trait MinLengthStdDevParams extends Params {
  final val minLengthStdDev = new DoubleParam(
    parent = this, name = "minLengthStdDev",
    doc = "minimum standard deviation of the lengths of tokens in a text field for it to be hashed instead " +
      "of ignored",
    isValid = ParamValidators.inRange(lowerBound = 0, upperBound = 100)
  )
  final def setMinLengthStdDev(v: Double): this.type = set(minLengthStdDev, v)
  final def getMinLengthStdDev: Double = $(minLengthStdDev)
  setDefault(minLengthStdDev -> SmartTextVectorizer.MinTextLengthStdDev)

  final val textLengthType: Param[String] = new Param[String](this, "textLengthType",
    "Method to use to construct length distribution from text in TextStats. Current options are" +
      "FullEntry (lengths are of entire entry) or Tokens (lengths are token lengths).",
    (value: String) => TextLengthType.withNameInsensitiveOption(value).isDefined
  )
  def setTextLengthType(v: TextLengthType): this.type = set(textLengthType, v.entryName)
  def getTextLengthType: TextLengthType = TextLengthType.withNameInsensitive($(textLengthType))
  setDefault(textLengthType -> TextLengthType.FullEntry.entryName)
}
