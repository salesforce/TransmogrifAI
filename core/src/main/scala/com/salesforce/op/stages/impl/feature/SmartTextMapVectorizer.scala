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
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.SmartTextVectorizerAction._
import com.salesforce.op.stages.impl.feature.VectorizerUtils._
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.stages.SensitiveFeatureMode._
import com.salesforce.op.utils.stages.{NameDetectFun, NameDetectStats}
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.macros.caseclass
import com.twitter.algebird.{Monoid, Semigroup}
import org.apache.spark.sql.{Dataset, Encoder, Encoders}

import scala.reflect.runtime.universe.TypeTag

/**
 * Convert a sequence of text map features into a vector by detecting categoricals that are disguised as text.
 * A categorical will be represented as a vector consisting of occurrences of top K most common values of that feature
 * plus occurrences of non top k values and a null indicator (if enabled).
 * Non-categoricals will be converted into a vector using the hashing trick. In addition, a null indicator is created
 * for each non-categorical (if enabled).
 *
 * Detection and removal of names in the input columns can be enabled with the `sensitiveFeatureMode` param.
 *
 * @param uid uid for instance
 */
class SmartTextMapVectorizer[T <: OPMap[String]]
(
  uid: String = UID[SmartTextMapVectorizer[T]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends SequenceEstimator[T, OPVector](operationName = "smartTxtMapVec", uid = uid)
    with PivotParams with CleanTextFun with SaveOthersParams
    with TrackNullsParam with MinSupportParam with TextTokenizerParams with TrackTextLenParam
    with HashingVectorizerParams with MapHashingFun with OneHotFun with MapStringPivotHelper
    with MapVectorizerFuns[String, OPMap[String]] with MaxCardinalityParams
    with NameDetectFun[Text] {

  private implicit val textMapStatsSeqEnc: Encoder[Array[TextMapStats]] = Encoders.kryo[Array[TextMapStats]]

  private def computeTextMapStats
  (
    textMap: T#Value,
    shouldCleanKeys: Boolean,
    shouldCleanValues: Boolean,
    nameDetectMapFun: Text#Value => NameDetectStats
  ): TextMapStats = {
    val keyValueCounts = textMap.map{ case (k, v) =>
      cleanTextFn(k, shouldCleanKeys) -> TextStats(Map(cleanTextFn(v, shouldCleanValues) -> 1))
    }
    val nameDetectStats = if (getSensitiveFeatureMode == Off) Map.empty[String, NameDetectStats]
    else textMap.map{ case (k, v) => cleanTextFn(k, shouldCleanKeys) -> nameDetectMapFun(Text(v).value) }
    TextMapStats(keyValueCounts, nameDetectStats)
  }

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

  private def makeVectorMetadata(
    args: SmartTextMapVectorizerModelArgs,
    aggNameDetectStats: Array[Map[String, NameDetectStats]]
  ): OpVectorMetadata = {
    val categoricalColumns = if (args.categoricalFeatureInfo.flatten.nonEmpty) {
      val (mapFeatures, mapFeatureInfo) =
        inN.toSeq.zip(args.categoricalFeatureInfo).filter{ case (tf, featureInfoSeq) => featureInfoSeq.nonEmpty }.unzip
      val topValues = mapFeatureInfo.map(featureInfoSeq =>
        featureInfoSeq.map(featureInfo => featureInfo.key -> featureInfo.topValues)
      )
      makeVectorColumnMetadata(
        topValues = topValues,
        inputFeatures = mapFeatures.toArray,
        unseenName = $(unseenName),
        trackNulls = args.shouldTrackNulls
      )
    } else Array.empty[OpVectorColumnMetadata]

    val textColumns = if (args.textFeatureInfo.flatten.nonEmpty) {
      val (mapFeatures, mapFeatureInfo) =
        inN.toSeq.zip(args.textFeatureInfo).filter{ case (tf, featureInfoSeq) => featureInfoSeq.nonEmpty }.unzip
      val allKeys = mapFeatureInfo.map(_.map(_.key))
      makeVectorColumnMetadata(
        features = mapFeatures.toArray,
        params = makeHashingParams(),
        allKeys = allKeys,
        shouldTrackNulls = args.shouldTrackNulls,
        shouldTrackLen = $(trackTextLen)
      )
    } else Array.empty[OpVectorColumnMetadata]
    val columns = categoricalColumns ++ textColumns

    val nameDetectStatsMap: Map[(String, Option[String]), NameDetectStats] =
      inN.toSeq.zip(aggNameDetectStats) flatMap {
        case (tf, mapFromKeyToNameDetectStats) => mapFromKeyToNameDetectStats map {
          case (key, nameDetectStats) => (tf.name, Some(key)) -> nameDetectStats }
      } toMap
    val sensitive = createSensitiveFeatureInformation(nameDetectStatsMap)

    OpVectorMetadata(getOutputFeatureName, columns, Transmogrifier.inputFeaturesToHistory(inN, stageName), sensitive)
  }

  def makeSmartTextMapVectorizerModelArgs(aggregatedStats: Array[TextMapStats]): SmartTextMapVectorizerModelArgs = {
    val maxCard = $(maxCardinality)
    val minSup = $(minSupport)
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = $(cleanText)
    val shouldTrackNulls = $(trackNulls)

    val allFeatureInfo = aggregatedStats.toSeq.map { textMapStats =>
      val featureInfoBeforeSensitive = textMapStats.keyValueCounts.toSeq.map { case (k, textStats) =>
        val whichAction = if (textStats.valueCounts.size <= maxCard) Categorical else NonCategorical
        val topVals = if (whichAction == Categorical) {
          textStats.valueCounts
            .filter { case (_, count) => count >= minSup }
            .toSeq.sortBy(v => -v._2 -> v._1)
            .take($(topK)).map(_._1).toArray
        } else Array.empty[String]
        SmartTextFeatureInfo(key = k, whichAction = whichAction, topValues = topVals)
      }

      if (shouldRemoveSensitive) textMapStats.nameDetectStats.toSeq.zip(featureInfoBeforeSensitive) map {
        case ((k, nameDetectStats), previousFeatureInfo) =>
        val treatAsName = computeTreatAsName(nameDetectStats)
        SmartTextFeatureInfo(key = k,
          whichAction = if (treatAsName) Sensitive else previousFeatureInfo.whichAction,
          topValues = previousFeatureInfo.topValues
        )
      } else featureInfoBeforeSensitive
    }

    SmartTextMapVectorizerModelArgs(
      allFeatureInfo = allFeatureInfo,
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues,
      shouldTrackNulls = shouldTrackNulls,
      hashingParams = makeHashingParams()
    )
  }

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    require(!dataset.isEmpty, "Input dataset cannot be empty")

    val maxCard = $(maxCardinality)
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = $(cleanText)

    implicit val textStatsMonoid: Monoid[TextMapStats] = TextMapStats.monoid(maxCard)
    val nameDetectMapFun = makeMapFunction(dataset.sparkSession)

    val valueStats: Dataset[Array[TextMapStats]] = dataset.map(
      _.map(computeTextMapStats(_, shouldCleanKeys, shouldCleanValues, nameDetectMapFun)).toArray
    )
    val aggregatedStats: Array[TextMapStats] = valueStats.reduce(_ + _)

    // TODO: Delete the following debugging tests
    println(
      aggregatedStats.flatMap(_.nameDetectStats.toSeq.map { case (k, v) => f"$k -> ${v.toString}" })
        .mkString(",")
    )
    import dataset.sparkSession.implicits._
    dataset.map(_.map(
      _.toSeq.map { case (k, v) => k -> (v, preProcess(Text(v))) }
    )).show(truncate = false)

    val smartTextMapVectorizerModelArgs = makeSmartTextMapVectorizerModelArgs(aggregatedStats)

    val vecMetadata = makeVectorMetadata(smartTextMapVectorizerModelArgs, aggregatedStats.map(_.nameDetectStats))
    setMetadata(vecMetadata.toMetadata)

    new SmartTextMapVectorizerModel[T](args = smartTextMapVectorizerModelArgs, operationName = operationName, uid = uid)
      .setAutoDetectLanguage(getAutoDetectLanguage)
      .setAutoDetectThreshold(getAutoDetectThreshold)
      .setDefaultLanguage(getDefaultLanguage)
      .setMinTokenLength(getMinTokenLength)
      .setToLowercase(getToLowercase)
      .setTrackTextLen($(trackTextLen))
  }
}

/**
 * Summary statistics of a text feature
 *
 * @param keyValueCounts counts of feature values
 */
private[op] case class TextMapStats
(
  keyValueCounts: Map[String, TextStats],
  nameDetectStats: Map[String, NameDetectStats] = Map.empty[String, NameDetectStats]
) extends JsonLike

private[op] object TextMapStats {

  def monoid(maxCardinality: Int): Monoid[TextMapStats] = {
    implicit val textStatsMonoid: Semigroup[TextStats] = TextStats.monoid(maxCardinality)
    implicit val nameDetectStatsMonoid: Semigroup[NameDetectStats] = NameDetectStats.monoid
    caseclass.monoid[TextMapStats]
  }

}

/**
 * Info about each feature within a text map
 *
 * @param key           name of a feature
 * @param whichAction indicate whether a feature is categorical or not
 * @param topValues     most common values of a feature (only for categoricals)
 */
case class SmartTextFeatureInfo
(
  key: String,
  whichAction: SmartTextVectorizerAction,
  topValues: Array[String]
) extends JsonLike


/**
 * Arguments for [[SmartTextMapVectorizerModel]]
 *
 * @param allFeatureInfo    info about each feature with each text map
 * @param shouldCleanKeys   should clean feature keys
 * @param shouldCleanValues should clean feature values
 * @param shouldTrackNulls  should track nulls
 * @param hashingParams     hashing function params
 */
case class SmartTextMapVectorizerModelArgs
(
  allFeatureInfo: Seq[Seq[SmartTextFeatureInfo]],
  shouldCleanKeys: Boolean,
  shouldCleanValues: Boolean,
  shouldTrackNulls: Boolean,
  hashingParams: HashingFunctionParams
) extends JsonLike {
  val (categoricalFeatureInfo, textFeatureInfo) = allFeatureInfo.map{ featureInfoSeq =>
    featureInfoSeq
      .filter { _.whichAction != Sensitive }
      .partition { _.whichAction == Categorical }
  }.unzip
  val categoricalKeys: Seq[Seq[String]] = categoricalFeatureInfo.map(featureInfoSeq => featureInfoSeq.map(_.key))
  val textKeys: Seq[Seq[String]] = textFeatureInfo.map(featureInfoSeq => featureInfoSeq.map(_.key))
}


final class SmartTextMapVectorizerModel[T <: OPMap[String]] private[op]
(
  val args: SmartTextMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T]) extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
  with TextTokenizerParams
  with TrackTextLenParam
  with MapHashingFun
  with TextMapPivotVectorizerModelFun[OPMap[String]] {

  private val categoricalPivotFn = pivotFn(
    topValues = args.categoricalFeatureInfo.filter(_.nonEmpty).map(_.map(info => info.key -> info.topValues)),
    shouldCleanKeys = args.shouldCleanKeys,
    shouldCleanValues = args.shouldCleanValues,
    shouldTrackNulls = args.shouldTrackNulls
  )

  private def partitionRow(row: Seq[OPMap[String]]):
  (Seq[OPMap[String]], Seq[Seq[String]], Seq[OPMap[String]], Seq[Seq[String]]) = {
    val (rowCategorical, keysCategorical) =
      row.view.zip(args.categoricalKeys).collect { case (elements, keys) if keys.nonEmpty =>
        val filtered = elements.value.filter { case (k, v) => keys.contains(k) }
        (TextMap(filtered), keys)
      }.unzip

    val (rowText, keysText) =
      row.view.zip(args.textKeys).collect { case (elements, keys) if keys.nonEmpty =>
        val filtered = elements.value.filter { case (k, v) => keys.contains(k) }
        (TextMap(filtered), keys)
      }.unzip

    (rowCategorical.toList, keysCategorical.toList, rowText.toList, keysText.toList)
  }

  def transformFn: Seq[T] => OPVector = row => {
    val (rowCategorical, keysCategorical, rowText, keysText) = partitionRow(row)
    val categoricalVector = categoricalPivotFn(rowCategorical)
    val rowTextTokenized = rowText.map(_.value.map { case (k, v) => k -> tokenize(v.toText).tokens })
    val textVector = hash(rowTextTokenized, keysText, args.hashingParams)
    val textNullIndicatorsVector =
      if (args.shouldTrackNulls) getNullIndicatorsVector(keysText, rowTextTokenized) else OPVector.empty
    val textLenVector = if ($(trackTextLen)) getLenVector(keysText, rowTextTokenized) else OPVector.empty

    categoricalVector.combine(textVector, textLenVector, textNullIndicatorsVector)
  }

  private def getNullIndicatorsVector(keysSeq: Seq[Seq[String]], inputs: Seq[Map[String, TextList]]): OPVector = {
    val nullIndicators = keysSeq.zip(inputs).flatMap{ case (keys, input) =>
      keys.map{ k =>
        val nullVal = if (input.get(k).forall(_.isEmpty)) 1.0 else 0.0
        Seq(0 -> nullVal)
      }
    }
    val reindexed = reindex(nullIndicators)
    val vector = makeSparseVector(reindexed)
    vector.toOPVector
  }

  private def getLenVector(keysSeq: Seq[Seq[String]], inputs: Seq[Map[String, TextList]]): OPVector = {
    keysSeq.zip(inputs).flatMap{ case (keys, input) =>
      keys.map(k => input.getOrElse(k, TextList.empty).value.map(_.length).sum.toDouble)
    }.toOPVector
  }
}

