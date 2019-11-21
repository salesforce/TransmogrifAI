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

import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.VectorizerUtils._
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.stages.NameIdentificationFun
import com.salesforce.op.utils.stages.NameIdentificationUtils._
import com.salesforce.op.{SensitiveFeatureInformation, UID}
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.{Monoid, Semigroup}
import enumeratum.EnumEntry
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Encoder}

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
 * @param uid           uid for instance
 * @param operationName unique name of the operation this stage performs
 * @param tti           type tag for input
 * @tparam T
 */
class SmartTextVectorizer[T <: Text]
(
  uid: String = UID[SmartTextVectorizer[T]],
  operationName: String = "smartTxtVec"
)(implicit tti: TypeTag[T]) extends SequenceEstimator[T, OPVector](
  uid = uid,
  operationName = operationName
) with PivotParams with CleanTextFun with SaveOthersParams
  with TrackNullsParam with MinSupportParam with TextTokenizerParams with TrackTextLenParam
  with HashingVectorizerParams with HashingFun with OneHotFun with MaxCardinalityParams
  with BiasDetectionParams with NameIdentificationFun[T] {

  private implicit val textStatsSeqEnc: Encoder[Array[TextStats]] = ExpressionEncoder[Array[TextStats]]()

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

  /* CODE FOR DETECTING SENSITIVE FEATURES BEGIN */
  var guardCheckResults: Option[Array[Boolean]] = None

  override def fit(dataset: Dataset[_]): SequenceModel[T, OPVector] = {
    /* Set instance variable for guardCheck results here.
    We compute guardChecks here because all of the logical checks can be implemented efficiently in native Spark,
    and we would like to use the individual Text columns before they are combined in SequenceEstimator.fit().
    It is not directly possible to do this computation in the treeAggregate() call below in fitFn() because one of the
    logical checks computes a standard deviation, which requires knowing the mean beforehand. We could implement the
    functionality in treeAggregate() if we could replace the standard deviation computation. */
    if (getRemoveSensitive) {
      guardCheckResults = Some(
        inN.map(feature => guardChecks(dataset.asInstanceOf[Dataset[T#Value]], col(feature.name)))
      )
    }
    // then call super
    super.fit(dataset)
  }

  private def detectSensitive(dataset: Dataset[Seq[T#Value]]): NameIdentificationResults = {
    val spark = dataset.sparkSession
    val broadcastNameDict: Broadcast[NameDictionary] = spark.sparkContext.broadcast(NameDictionary())
    val broadcastGenderDict: Broadcast[GenderDictionary] = spark.sparkContext.broadcast(GenderDictionary())

    def aggregateTwoResults(
      one: NameIdentificationAccumulator, two: NameIdentificationAccumulator
    ): NameIdentificationAccumulator = {
      NameIdentificationAccumulator(
        one.count + two.count,
        one.predictedNameProb + two.predictedNameProb,
        one.tokenInFirstNameDictionary + two.tokenInFirstNameDictionary,
        one.tokenIsMale + two.tokenIsMale,
        one.tokenIsFemale + two.tokenIsFemale,
        one.tokenIsOther + two.tokenIsOther
      )
    }

    def aggregateSeqResults(
      one: Seq[NameIdentificationAccumulator], two: Seq[NameIdentificationAccumulator]
    ): Seq[NameIdentificationAccumulator] = {
      one zip two map { case (x, y) => aggregateTwoResults(x, y) }
    }

    import com.salesforce.op.features.types.NameStats.GenderStrings._
    def computeResults(input: T#Value): NameIdentificationAccumulator = {
      if (input.isEmpty) NameIdentificationAccumulator()
      else {
        val tokens: Seq[String] = preProcess(input)
        val (firstHalf, secondHalf) = (TokensToCheckForFirstName map { index: Int =>
          val (inFirstNameDict, isMale, isFemale, isOther) = identifyGender(tokens, index, broadcastGenderDict) match {
            case Male => (1, 1, 0, 0)
            case Female => (1, 0, 1, 0)
            case _ => (0, 0, 0, 1)
          }
          ((index -> inFirstNameDict, index -> isMale), (index -> isFemale, index -> isOther))
        }).unzip
        val (inFirstNameDictSeq, isMaleSeq) = firstHalf.unzip
        val (isFemaleSeq, isOtherSeq) = secondHalf.unzip
        NameIdentificationAccumulator(
          1.0,
          dictCheck(tokens, broadcastNameDict),
          Map(inFirstNameDictSeq: _*),
          Map(isMaleSeq: _*),
          Map(isFemaleSeq: _*),
          Map(isOtherSeq: _*)
        )
      }
    }

    val rdd = dataset.rdd
    val zeroValue = Seq.fill[NameIdentificationAccumulator](inN.length)(NameIdentificationAccumulator())
    val agg = rdd.treeAggregate[Seq[NameIdentificationAccumulator]](zeroValue)(
      combOp = aggregateSeqResults, seqOp = {
        case (result, row) => aggregateSeqResults(result, row.map(computeResults))
      }
    ).toArray

    val nonNullCounts = agg map { _.count }
    def normalize(arr: Seq[Double]): Seq[Double] = arr zip nonNullCounts map { case (x, count) => x / count }

    val predictedProbs = normalize(agg map { _.predictedNameProb }).toArray
    val isName = guardCheckResults match {
      case Some(results) => predictedProbs zip results map {
        case (prob, guardCheck) => guardCheck && prob >= $(defaultThreshold)
      }
      case _ =>
        throw new RuntimeException("Guard check results were not generated but this should not happen.")
    }
    val bestFirstNameIndexes: Array[Int] = agg map { result: NameIdentificationAccumulator =>
      val (bestIndex, _) = if (result.tokenInFirstNameDictionary.isEmpty) (0, 0)
      else result.tokenInFirstNameDictionary.maxBy(_._2)
      bestIndex
    }
    val (numMale, numFemale, numOther) = (agg zip bestFirstNameIndexes).map {
      case (result: NameIdentificationAccumulator, index: Int) =>
      (
        result.tokenIsMale.getOrElse(index, 0).toDouble,
        result.tokenIsFemale.getOrElse(index, 0).toDouble,
        result.tokenIsOther.getOrElse(index, 0).toDouble
      )
    }.unzip3
    val (pctMale, pctFemale, pctOther) = (
      normalize(numMale).toArray, normalize(numFemale).toArray, normalize(numOther).toArray
    )
    NameIdentificationResults(isName, predictedProbs, bestFirstNameIndexes, pctMale, pctFemale, pctOther)
  }
  /* CODE FOR DETECTING SENSITIVE FEATURES END */

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    require(!dataset.isEmpty, "Input dataset cannot be empty")

    val maxCard = $(maxCardinality)
    val shouldCleanText = $(cleanText)

    implicit val testStatsMonoid: Semigroup[TextStats] = TextStats.monoid(maxCard)
    val valueStats: Dataset[Array[TextStats]] = dataset.map(_.map(computeTextStats(_, shouldCleanText)).toArray)
    val aggregatedStats: Array[TextStats] = valueStats.reduce(_ + _)

    val (isCategorical, topValues) = aggregatedStats.map { stats =>
      val isCategorical = stats.valueCounts.size <= maxCard
      val topValues = stats.valueCounts
        .filter { case (_, count) => count >= $(minSupport) }
        .toSeq.sortBy(v => -v._2 -> v._1)
        .take($(topK)).map(_._1)
      isCategorical -> topValues
    }.unzip

    val nameIdentificationResults = if (getRemoveSensitive) Some(detectSensitive(dataset)) else None

    val smartTextParams = SmartTextVectorizerModelArgs(
      isCategorical = isCategorical,
      topValues = topValues,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = $(trackNulls),
      hashingParams = makeHashingParams(),
      isName = nameIdentificationResults.map(_.isName).getOrElse(Array.empty[Boolean]),
      removeSensitive = getRemoveSensitive
    )

    val vecMetadata: OpVectorMetadata = makeVectorMetadata(smartTextParams, nameIdentificationResults)
    setMetadata(vecMetadata.toMetadata)

    new SmartTextVectorizerModel[T](args = smartTextParams, operationName = operationName, uid = uid)
      .setAutoDetectLanguage(getAutoDetectLanguage)
      .setAutoDetectThreshold(getAutoDetectThreshold)
      .setDefaultLanguage(getDefaultLanguage)
      .setMinTokenLength(getMinTokenLength)
      .setToLowercase(getToLowercase)
      .setTrackTextLen($(trackTextLen))
  }

  private def computeTextStats(text: T#Value, shouldCleanText: Boolean): TextStats = {
    val valueCounts = text match {
      case Some(v) => Map(cleanTextFn(v, shouldCleanText) -> 1)
      case None => Map.empty[String, Int]
    }
    TextStats(valueCounts)
  }

  protected def makeVectorMetadata(
    smartTextParams: SmartTextVectorizerModelArgs,
    nameIdentificationResults: Option[NameIdentificationResults]
  ): OpVectorMetadata = {
    require(inN.length == smartTextParams.isCategorical.length)

    val (features, isCategorical): (Array[TransientFeature], Array[Boolean]) = nameIdentificationResults match {
      case Some(results) if getRemoveSensitive =>
        (
          SmartTextVectorizer.partition[TransientFeature](inN, results.isName)._2,
          SmartTextVectorizer.partition[Boolean](smartTextParams.isCategorical, results.isName)._2
        )
      case _ => (inN, smartTextParams.isCategorical)
    }

    val (categoricalFeatures, textFeatures) =
      SmartTextVectorizer.partition[TransientFeature](features, isCategorical)

    // build metadata describing output
    val shouldTrackNulls = $(trackNulls)
    val shouldTrackLen = $(trackTextLen)
    val unseen = Option($(unseenName))

    val categoricalColumns = if (categoricalFeatures.nonEmpty) {
      makeVectorColumnMetadata(shouldTrackNulls, unseen, smartTextParams.categoricalTopValues, categoricalFeatures)
    } else Array.empty[OpVectorColumnMetadata]
    val textColumns = if (textFeatures.nonEmpty) {
      if (shouldTrackLen) {
        makeVectorColumnMetadata(textFeatures, makeHashingParams()) ++
          textFeatures.map(_.toColumnMetaData(descriptorValue = OpVectorColumnMetadata.TextLenString)) ++
          textFeatures.map(_.toColumnMetaData(isNull = true))
      }
      else {
        makeVectorColumnMetadata(textFeatures, makeHashingParams()) ++
          textFeatures.map(_.toColumnMetaData(isNull = true))
      }
    } else Array.empty[OpVectorColumnMetadata]

    val columns = categoricalColumns ++ textColumns

    val sensitive: Map[String, SensitiveFeatureInformation] =
      nameIdentificationResults map { results: NameIdentificationResults =>
        logDebug(s"treatAsName: [${results.isName.mkString(",")}]")
        logDebug(s"predictedNameProb: [${results.predictedNameProbs.mkString(",")}]")
        logDebug(s"bestFirstNameIndexes: [${results.bestFirstNameIndexes.mkString(",")}]")
        logDebug(s"pctMale: [${results.pctMale.mkString(",")}]")
        logDebug(s"pctFemale: [${results.pctFemale.mkString(",")}]")
        logDebug(s"pctOther: [${results.pctOther.mkString(",")}]")

        if (results.isName exists identity) {
          logWarning(
            """Hey! Some of your text columns look like they have names in them. Are you sure you want to build a
              |model with that information in them? There could be potential bias as a result! Here's what I found:
              |""".stripMargin
          )
          val problemColIndexes = results.isName.zipWithIndex.filter(_._1).map(_._2)
          problemColIndexes foreach { index: Int =>
            logWarning {
              s"""Column Name: ${inN(index).name}
              |Predicted Probability of Name: ${results.predictedNameProbs(index)}
              |Percentage Likely Male Names: ${results.pctMale(index)}
              |Percentage Likely Female Names: ${results.pctFemale(index)}
              |Percentage Where No Gender Found: ${results.pctOther(index)}
              |""".stripMargin
            }
          }
        }

        // Transform into the tuple from feature name to SensitiveFeatureInformation
        inN.zipWithIndex map { case (feature: TransientFeature, index: Int) =>
          feature.name -> SensitiveFeatureInformation.Name(
            getRemoveSensitive && results.isName(index),
            results.predictedNameProbs(index),
            Seq.empty[String], // TODO: Keep track of the detected names here
            results.pctMale(index),
            results.pctFemale(index),
            results.pctOther(index)
          )
        } toMap
    } getOrElse Map.empty[String, SensitiveFeatureInformation]
    OpVectorMetadata(getOutputFeatureName, columns, Transmogrifier.inputFeaturesToHistory(inN, stageName), sensitive)
  }
}

object SmartTextVectorizer {
  val MaxCardinality = 100
  private[op] def partition[T: ClassTag](input: Array[T], condition: Array[Boolean]): (Array[T], Array[T]) = {
    val all = input.zip(condition)
    (all.collect { case (item, true) => item }.toSeq.toArray, all.collect { case (item, false) => item }.toSeq.toArray)
  }
}

/**
 * Summary statistics of a text feature
 *
 * @param valueCounts counts of feature values
 */
private[op] case class TextStats(valueCounts: Map[String, Int]) extends JsonLike

private[op] object TextStats {
  def monoid(maxCardinality: Int): Monoid[TextStats] = new Monoid[TextStats] {
    override def plus(l: TextStats, r: TextStats): TextStats = {
      if (l.valueCounts.size > maxCardinality) l
      else if (r.valueCounts.size > maxCardinality) r
      else TextStats(l.valueCounts + r.valueCounts)
    }

    override def zero: TextStats = TextStats.empty
  }

  def empty: TextStats = TextStats(Map.empty)
}

import enumeratum._
sealed trait SensitiveFeatureMode extends EnumEntry with Serializable
object SensitiveFeatureMode extends Enum[SensitiveFeatureMode] {
  val values = findValues

  case object Off extends SensitiveFeatureMode
  case object DetectOnly extends SensitiveFeatureMode
  case object DetectAndRemove extends SensitiveFeatureMode
}

/**
 * Arguments for [[SmartTextVectorizerModel]]
 *
 * @param isCategorical       is feature a categorical or not
 * @param topValues           top values to each feature
 * @param shouldCleanText     should clean text value
 * @param shouldTrackNulls    should track nulls
 * @param hashingParams       hashing function params
 * @param removeSensitive     whether to remove detected sensitive fields
 */
case class SmartTextVectorizerModelArgs
(
  isCategorical: Array[Boolean],
  topValues: Array[Seq[String]],
  shouldCleanText: Boolean,
  shouldTrackNulls: Boolean,
  hashingParams: HashingFunctionParams,
  isName: Array[Boolean],
  removeSensitive: Boolean = false
) extends JsonLike {
  def categoricalTopValues: Array[Seq[String]] =
    topValues.zip(isCategorical).collect { case (top, true) => top }
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
    row: Seq[Text] => {
      val (cleanedRow, isCategorical) = if (args.removeSensitive) {
        (
          SmartTextVectorizer.partition[Text](row.toArray, args.isName)._2,
          SmartTextVectorizer.partition[Boolean](args.isCategorical, args.isName)._2
        )
      } else (row.toArray, args.isCategorical)
      val (rowCategorical, rowText) = SmartTextVectorizer.partition[Text](cleanedRow, isCategorical)

      val categoricalVector: OPVector = categoricalPivotFn(rowCategorical)
      val textTokens: Seq[TextList] = rowText.map(tokenize(_).tokens)

      val textVector: OPVector = hash[TextList](textTokens, getTextTransientFeatures, args.hashingParams)
      val textNullIndicatorsVector = if (args.shouldTrackNulls) getNullIndicatorsVector(textTokens) else OPVector.empty
      val textLenVector = if ($(trackTextLen)) getLenVector(textTokens) else OPVector.empty

      categoricalVector.combine(textVector, textLenVector, textNullIndicatorsVector)
    }
  }

  private def getTextTransientFeatures: Array[TransientFeature] = {
    val nonNameIsCategorical = if (args.removeSensitive) {
      SmartTextVectorizer.partition[Boolean](args.isCategorical, args.isName)._2
    } else args.isCategorical
    SmartTextVectorizer.partition[TransientFeature](getTransientFeatures(), nonNameIsCategorical)._2
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

/* CODE FOR DETECTING SENSITIVE FEATURES BEGIN */
trait BiasDetectionParams extends Params {
  final val sensitiveFeatureMode: Param[String] = new Param[String](this, "sensitiveFeatureMode",
    "Whether to detect sensitive features and how to handle them",
    (value: String) => SensitiveFeatureMode.withNameInsensitiveOption(value).isDefined
  )
  setDefault(sensitiveFeatureMode, SensitiveFeatureMode.Off.toString)
  def setSensitiveFeatureMode(v: SensitiveFeatureMode): this.type = set(sensitiveFeatureMode, v.entryName)
  def getSensitiveFeatureMode: SensitiveFeatureMode = SensitiveFeatureMode.withNameInsensitive($(sensitiveFeatureMode))
  def getRemoveSensitive: Boolean = {
    SensitiveFeatureMode.withNameInsensitive($(sensitiveFeatureMode)) == SensitiveFeatureMode.DetectAndRemove
  }

  final val defaultThreshold = new DoubleParam(
    parent = this,
    name = "defaultThreshold",
    doc = "default fraction of entries to be names before treating as name",
    isValid = (value: Double) => {
      ParamValidators.gt(0.0)(value) && ParamValidators.lt(1.0)(value)
    }
  )
  setDefault(defaultThreshold, 0.50)
  def setThreshold(value: Double): this.type = set(defaultThreshold, value)
}

/**
 * Case class for gathering results in the Spark dataset during treeAggregate
 * @param count
 * @param predictedNameProb
 * @param tokenInFirstNameDictionary
 * @param tokenIsMale
 * @param tokenIsFemale
 * @param tokenIsOther
 */
case class NameIdentificationAccumulator
(
  count: Double = 0.0,
  predictedNameProb: Double = 0.0,
  tokenInFirstNameDictionary: Map[Int, Int] = EmptyTokensMap,
  tokenIsMale: Map[Int, Int] = EmptyTokensMap,
  tokenIsFemale: Map[Int, Int] = EmptyTokensMap,
  tokenIsOther: Map[Int, Int] = EmptyTokensMap
) extends JsonLike

/**
 * Case class for collecting the overall results from the name identification step
 * @param isName
 * @param predictedNameProbs
 * @param bestFirstNameIndexes
 * @param pctMale
 * @param pctFemale
 * @param pctOther
 */
case class NameIdentificationResults
(
  isName: Array[Boolean],
  predictedNameProbs: Array[Double],
  bestFirstNameIndexes: Array[Int],
  pctMale: Array[Double],
  pctFemale: Array[Double],
  pctOther: Array[Double]
) extends JsonLike
/* CODE FOR DETECTING SENSITIVE FEATURES END */
