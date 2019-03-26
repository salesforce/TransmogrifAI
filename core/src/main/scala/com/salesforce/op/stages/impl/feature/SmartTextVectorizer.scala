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
import com.salesforce.op.features.types.{OPVector, Text, TextList, VectorConversions, SeqDoubleConversions}
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.VectorizerUtils._
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.Semigroup
import org.apache.spark.ml.param._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
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
 * @param uid uid for instance
 */
class SmartTextVectorizer[T <: Text](uid: String = UID[SmartTextVectorizer[T]])(implicit tti: TypeTag[T])
  extends SequenceEstimator[T, OPVector](operationName = "smartTxtVec", uid = uid)
    with PivotParams with CleanTextFun with SaveOthersParams
    with TrackNullsParam with MinSupportParam with TextTokenizerParams with TrackTextLenParam
    with HashingVectorizerParams with HashingFun with OneHotFun with MaxCardinalityParams {

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

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    require(!dataset.isEmpty, "Input dataset cannot be empty")

    val maxCard = $(maxCardinality)
    val shouldCleanText = $(cleanText)

    implicit val testStatsSG: Semigroup[TextStats] = TextStats.semiGroup(maxCard)
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

    val smartTextParams = SmartTextVectorizerModelArgs(
      isCategorical = isCategorical,
      topValues = topValues,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = $(trackNulls),
      hashingParams = makeHashingParams()
    )

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

  private def computeTextStats(text: T#Value, shouldCleanText: Boolean): TextStats = {
    val valueCounts = text match {
      case Some(v) => Map(cleanTextFn(v, shouldCleanText) -> 1)
      case None => Map.empty[String, Int]
    }
    TextStats(valueCounts)
  }

  private def makeVectorMetadata(smartTextParams: SmartTextVectorizerModelArgs): OpVectorMetadata = {
    require(inN.length == smartTextParams.isCategorical.length)

    val (categoricalFeatures, textFeatures) =
      SmartTextVectorizer.partition[TransientFeature](inN, smartTextParams.isCategorical)

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
    OpVectorMetadata(getOutputFeatureName, columns, Transmogrifier.inputFeaturesToHistory(inN, stageName))
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
  def semiGroup(maxCardinality: Int): Semigroup[TextStats] = new Semigroup[TextStats] {
    override def plus(l: TextStats, r: TextStats): TextStats = {
      if (l.valueCounts.size > maxCardinality) l
      else if (r.valueCounts.size > maxCardinality) r
      else TextStats(l.valueCounts + r.valueCounts)
    }
  }

  def empty: TextStats = TextStats(Map.empty)
}

/**
 * Arguments for [[SmartTextVectorizerModel]]
 *
 * @param isCategorical    is feature a categorical or not
 * @param topValues        top values to each feature
 * @param shouldCleanText  should clean text value
 * @param shouldTrackNulls should track nulls
 * @param hashingParams    hashing function params
 */
case class SmartTextVectorizerModelArgs
(
  isCategorical: Array[Boolean],
  topValues: Array[Seq[String]],
  shouldCleanText: Boolean,
  shouldTrackNulls: Boolean,
  hashingParams: HashingFunctionParams
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
    (row: Seq[Text]) => {
      val (rowCategorical, rowText) = SmartTextVectorizer.partition[Text](row.toArray, args.isCategorical)
      val categoricalVector: OPVector = categoricalPivotFn(rowCategorical)
      val textTokens: Seq[TextList] = rowText.map(tokenize(_).tokens)
      val textVector: OPVector = hash[TextList](textTokens, getTextTransientFeatures, args.hashingParams)
      val textNullIndicatorsVector = if (args.shouldTrackNulls) getNullIndicatorsVector(textTokens) else OPVector.empty
      val textLenVector = if ($(trackTextLen)) getLenVector(textTokens) else OPVector.empty

      categoricalVector.combine(textVector, textLenVector, textNullIndicatorsVector)
    }
  }

  private def getTextTransientFeatures: Array[TransientFeature] =
    SmartTextVectorizer.partition[TransientFeature](getTransientFeatures(), args.isCategorical)._2

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

trait MaxPercentageCardinalityParams extends Params {
  final val maxPercentageCardinality = new DoubleParam(
    parent = this, name = "maxPercentageCardinality",
    doc = "max percentage of distinct values a categorical feature can have",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0)
  )
  final def setMaxPercentageCardinality(v: Double): this.type = set(maxPercentageCardinality, v)
  final def getMaxPercentageCardinality: Double = $(maxPercentageCardinality)
  setDefault(maxPercentageCardinality -> 1.0)

  final val hllBits = new IntParam(
    parent = this, name = "hllBits", doc =
      "Number of bits used for hashing in HyperLogLog (HLL). Error is about 1.04/sqrt(2^{bits})." +
        " Default is 12 bits for 1% error which means each HLL instance is about 2^{12} = 4kb per instance.",
    isValid = ParamValidators.gtEq(4)
  )
  final def setHLLBits(value: Int): this.type = set(hllBits, value)
  final def getHLLBits: Int = $(hllBits)
  setDefault(hllBits, 12)
}
