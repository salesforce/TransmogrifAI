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
import com.salesforce.op.features.types.{OPVector, SeqDoubleConversions, Text, TextList, VectorConversions}
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.SmartTextVectorizer.hllBits
import com.salesforce.op.stages.impl.feature.VectorizerUtils._
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.twitter.algebird.{HLL, HyperLogLogMonoid, Monoid, Semigroup}
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import org.apache.spark.ml.param._
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
 * @param uid uid for instance
 */
class SmartTextVectorizer[T <: Text](uid: String = UID[SmartTextVectorizer[T]])(implicit tti: TypeTag[T])
  extends SequenceEstimator[T, OPVector](operationName = "smartTxtVec", uid = uid)
    with PivotParams with CleanTextFun with SaveOthersParams
    with TrackNullsParam with MinSupportParam with TextTokenizerParams with TrackTextLenParam
    with HashingVectorizerParams with HashingFun with OneHotFun with MaxCardinalityParams
    with MinLengthStdDevParams with AdaptiveHashParams
    with AdaptiveHashCollisionParams with MaxHashParams {

  private implicit val textStatsSeqEnc: Encoder[Array[TextStats]] = Encoders.kryo[Array[TextStats]]
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
    val shouldAdaptiveHash = $(adaptiveHash)
    val adaptiveHashCol = $(adaptiveHashCollision)
    val shouldTokenizeForLengths = $(textLengthType) == TextLengthType.Tokens.entryName
    val maxHashNum = $(maxHash)

    implicit val testStatsMonoid: Semigroup[TextStats] = TextStats.monoid(maxCard)
    val valueStats: Dataset[Array[TextStats]] = dataset.map(
      _.map(TextStats.computeTextStats(_, shouldCleanText, shouldTokenizeForLengths, maxCard)).toArray
    )
    val aggregatedStats: Array[TextStats] = valueStats.reduce(_ + _)

    val (vectorizationMethods, topValues, hashSizes) = aggregatedStats.map { stats =>
      val vecMethod: TextVectorizationMethod = stats match {
        case _ if stats.valueCounts.size <= maxCard => TextVectorizationMethod.Pivot
        case _ if stats.lengthStdDev < minLenStdDev => TextVectorizationMethod.Ignore
        case _ => TextVectorizationMethod.Hash
      }
      val topValues = stats.valueCounts
        .filter { case (_, count) => count >= $(minSupport) }
        .toSeq.sortBy(v => -v._2 -> v._1)
        .take($(topK)).map(_._1)

      val adaptiveHashSize = (stats.hll.estimatedSize / adaptiveHashCol).toInt

      (vecMethod, topValues, adaptiveHashSize)

    }.unzip3

    val numRawTextFeataure = vectorizationMethods.size
    val defaultHashSizes = Array.fill(numRawTextFeataure)($(numFeatures))
    val smartTextParams = SmartTextVectorizerModelArgs(
      vectorizationMethods = vectorizationMethods,
      topValues = topValues,
      adaptiveHashSizes = if (shouldAdaptiveHash) hashSizes else defaultHashSizes,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = $(trackNulls),
      hashingParams = makeHashingParams()
    )

    logInfo("TextStats for features used in SmartTextVectorizer:")
    inN.map(_.name).zip(aggregatedStats).zip(vectorizationMethods).foreach { case((name, stats), vecMethod) =>
      logInfo(s"Feature: $name")
      logInfo(s"LengthCounts: ${stats.lengthCounts}")
      logInfo(s"LengthMean: ${stats.lengthMean}")
      logInfo(s"LengthStdDev: ${stats.lengthStdDev}")
      logInfo(s"Vectorization method: $vecMethod")
      logInfo(s"Token cardinality: ${stats.tokenCard}")
      logInfo(s"Adaptive Hash size:" + (stats.tokenCard / adaptiveHashCol).toString)
    }

    val vecMetadata = makeVectorMetadata(smartTextParams)
    setMetadata(vecMetadata.toMetadata)

    new SmartTextVectorizerModel[T](args = smartTextParams, operationName = operationName, uid = uid)
      .setAutoDetectLanguage(getAutoDetectLanguage)
      .setAutoDetectThreshold(getAutoDetectThreshold)
      .setDefaultLanguage(getDefaultLanguage)
      .setMinTokenLength(getMinTokenLength)
      .setToLowercase(getToLowercase)
      .setTrackTextLen($(trackTextLen))
  }

  private def makeVectorMetadata(smartTextParams: SmartTextVectorizerModelArgs): OpVectorMetadata = {
    require(inN.length == smartTextParams.vectorizationMethods.length)

    val groups = inN.toArray
      .zip(smartTextParams.vectorizationMethods)
      .zip(smartTextParams.adaptiveHashSizes)
      .map{case ((a, b), c) => (a, b, c)}
      .groupBy(_._2)
    val textToPivot = groups.getOrElse(TextVectorizationMethod.Pivot, Array.empty).map(_._1)
    val textToIgnore = groups.getOrElse(TextVectorizationMethod.Ignore, Array.empty).map(_._1)
    val textToHash = groups.getOrElse(TextVectorizationMethod.Hash, Array.empty).map(_._1)
    val adaptiveHashSizes = groups.getOrElse(TextVectorizationMethod.Hash, Array.empty).map(_._3)
    val allTextFeatures = textToHash ++ textToIgnore

    // build metadata describing output
    val shouldTrackNulls = $(trackNulls)
    val shouldTrackLen = $(trackTextLen)
    val unseen = Option($(unseenName))

    val categoricalColumns = if (textToPivot.nonEmpty) {
      makeVectorColumnMetadata(shouldTrackNulls, unseen, smartTextParams.categoricalTopValues, textToPivot)
    } else Array.empty[OpVectorColumnMetadata]

    val textColumns = if (allTextFeatures.nonEmpty) {
      if (shouldTrackLen) {
        makeVectorColumnMetadata(textToHash, makeHashingParams(), Some(adaptiveHashSizes)) ++
          allTextFeatures.map(_.toColumnMetaData(descriptorValue = OpVectorColumnMetadata.TextLenString)) ++
          allTextFeatures.map(_.toColumnMetaData(isNull = true))
      }
      else {
        makeVectorColumnMetadata(textToHash, makeHashingParams(), Some(adaptiveHashSizes)) ++
          allTextFeatures.map(_.toColumnMetaData(isNull = true))
      }
    } else Array.empty[OpVectorColumnMetadata]

    val columns = categoricalColumns ++ textColumns
    OpVectorMetadata(getOutputFeatureName, columns, Transmogrifier.inputFeaturesToHistory(inN, stageName))
  }
}

object SmartTextVectorizer {
  val MaxCardinality: Int = 100
  val MinTextLengthStdDev: Double = 0
  val hllBits: Int = 12
  val AdaptiveHash: Boolean = false
  val AdaptiveHashCollision: Int = 20
  val LengthType: TextLengthType = TextLengthType.FullEntry
  val MaxHash: Int = 3000
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
private[op] case class TextStats
(
  valueCounts: Map[String, Long],
  lengthCounts: Map[Int, Long],
  hll: HLL
) extends JsonLike {

  val lengthSize = lengthCounts.values.sum
  val lengthMean: Double = lengthCounts.foldLeft(0.0)((acc, el) => acc + el._1 * el._2) / lengthSize
  val lengthVariance: Double = lengthCounts.foldLeft(0.0)(
    (acc, el) => acc + el._2 * (el._1 - lengthMean) * (el._1 - lengthMean)
  ) / lengthSize
  val lengthStdDev: Double = math.sqrt(lengthVariance)
  val tokenCard: Double = hll.estimatedSize
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
      val newHLL = l.hll + r.hll
      TextStats(newValueCounts, newLengthCounts, newHLL)
    }

    override def zero: TextStats = TextStats.empty
  }
  val hllMonoid = new HyperLogLogMonoid(hllBits)

  def empty: TextStats = TextStats(Map.empty, Map.empty, hllMonoid.zero)

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
      case None => TextStats.empty
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
    val cleanString = cleanTextFn(textString, shouldCleanText)
    val lengthsMap = if (shouldTokenize) {
      TextTokenizer.tokenizeString(textString).tokens.value
        .foldLeft(Map.empty[Int, Long])(
          (acc, el) => TextStats.additionHelper(acc, Map(el.length -> 1L), maxCardinality)
        )
    } else Map(cleanString.length -> 1L)
    val tokens = TextTokenizer.tokenizeString(textString).tokens.value
    val bigrams = (tokens zip tokens.drop(1)).map(t => t._1 + " " + t._2)
    val textHLL = bigrams
      .map(x => TextStats.hllMonoid.create(x.getBytes))
      .reduceOption(_ + _) match {
      case Some(x) => x
      case _ => TextStats.hllMonoid.zero
    }

    TextStats(Map(cleanTextFn(textString, shouldCleanText) -> 1L), lengthsMap, textHLL)
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
  adaptiveHashSizes: Array[Int],
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
      val groups = row.toArray
        .zip(args.vectorizationMethods)
        .zip(args.adaptiveHashSizes)
        .map{case ((a, b), c) => (a, b, c)}
        .groupBy(_._2)
      val textToPivot = groups.getOrElse(TextVectorizationMethod.Pivot, Array.empty).map(_._1)
      val textToIgnore = groups.getOrElse(TextVectorizationMethod.Ignore, Array.empty).map(_._1)
      val textToHash = groups.getOrElse(TextVectorizationMethod.Hash, Array.empty).map(_._1)
      val adaptiveHashSizes = groups.getOrElse(TextVectorizationMethod.Hash, Array.empty).map(_._3)
      val categoricalVector: OPVector = categoricalPivotFn(textToPivot)
      val textTokens: Seq[TextList] = textToHash.map(x => makeUniAndBigrams(tokenize(x).tokens))
      val ignorableTextTokens: Seq[TextList] = textToIgnore.map(tokenize(_).tokens)
      val textVector: OPVector = hash[TextList](
        textTokens, getTextTransientFeatures, args.hashingParams, Some(adaptiveHashSizes)
      )
      val textNullIndicatorsVector = if (args.shouldTrackNulls) {
        getNullIndicatorsVector(textTokens ++ ignorableTextTokens)
      } else OPVector.empty
      val textLenVector = if ($(trackTextLen)) getLenVector(textTokens ++ ignorableTextTokens) else OPVector.empty

      categoricalVector.combine(textVector, textLenVector, textNullIndicatorsVector)
    }
  }

  def makeUniAndBigrams(tl: TextList): TextList = {
    TextList(
      (tl.value zip tl.value.drop(1)).map(x => x._1 + " " + x._2)
    )
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

trait AdaptiveHashParams extends Params {
  final val adaptiveHash = new BooleanParam(
    parent = this, name = "adaptiveHash",
    doc = "If true we will adaptively set the hash space for each text feature, otherwise we will use the default"
  )
  final def setAdaptiveHash(v: Boolean): this.type = set(adaptiveHash, v)
  final def getAdaptiveHash: Boolean = $(adaptiveHash)
  setDefault(adaptiveHash -> SmartTextVectorizer.AdaptiveHash)
}

trait AdaptiveHashCollisionParams extends Params {
  final val adaptiveHashCollision = new IntParam(
    parent = this, name = "adaptiveHashCollision",
    doc = "If adaptiveHash is true we set tolerable threshold for hash collison"
  )
  final def setAdaptiveHashCollision(v: Int): this.type = set(adaptiveHashCollision, v)
  final def getAdaptiveHashCollision: Int = $(adaptiveHashCollision)
  setDefault(adaptiveHashCollision -> SmartTextVectorizer.AdaptiveHashCollision)
}

trait MaxHashParams extends Params {
  final val maxHash = new IntParam(
    parent = this, name = "maxHash",
    doc = "Max # of hash buckets"
  )
  final def setMaxHash(v: Int): this.type = set(maxHash, v)
  final def getMaxHash: Int = $(maxHash)
  setDefault(maxHash -> SmartTextVectorizer.MaxHash)
}
