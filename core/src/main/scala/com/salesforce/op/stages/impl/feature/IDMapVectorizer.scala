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

import com.salesforce.op.{FeatureHistory, UID}
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types.{ID, IDMap, OPMap, OPVector, TextMap}
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.Semigroup
import org.apache.spark.sql.{Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder

import scala.reflect.runtime.universe.TypeTag


class IDMapVectorizer(
  uid: String = UID[IDMapVectorizer]
)(implicit tti: TypeTag[IDMap], ttiv: TypeTag[IDMap#Value])
  extends SequenceEstimator[IDMap, OPVector](operationName = "dropIDMap", uid = uid)
    with PivotParams with CleanTextFun with SaveOthersParams
    with TrackNullsParam with MinSupportParam with OneHotFun
    with MapStringPivotHelper with MapVectorizerFuns[String, OPMap[String]] with MaxCardinalityParams {

  private implicit val textMapStatsSeqEnc: Encoder[Array[TextMapStats]] = ExpressionEncoder[Array[TextMapStats]]()

  private def computeTextMapStats
  (
    textMap: IDMap#Value, shouldCleanKeys: Boolean, shouldCleanValues: Boolean
  ): TextMapStats = {
    val keyValueCounts = textMap.map { case (k, v) =>
      cleanTextFn(k, shouldCleanKeys) -> TextStats(Map(cleanTextFn(v, shouldCleanValues) -> 1))
    }
    TextMapStats(keyValueCounts)
  }

  private def inputFeaturesToHistory(tf: Array[TransientFeature], thisStageName: String): Map[String, FeatureHistory] =
    tf.map(f => f.name -> FeatureHistory(originFeatures = f.originFeatures, stages = f.stages :+ thisStageName)).toMap

  private def makeVectorMetadata(args: IDMapVectorizerModelArgs): OpVectorMetadata = {
    val categoricalColumns = {
      val (mapFeatures, mapFeatureInfo) =
        inN.toSeq.zip(args.categoricalFeatureInfo).filter { case (tf, featureInfoSeq) => featureInfoSeq.nonEmpty }.unzip
      val topValues = mapFeatureInfo.map(featureInfoSeq =>
        featureInfoSeq.map(featureInfo => featureInfo.key -> featureInfo.topValues)
      )
      makeVectorColumnMetadata(
        topValues = topValues,
        inputFeatures = mapFeatures.toArray,
        unseenName = $(unseenName),
        trackNulls = args.shouldTrackNulls
      )
    }

    val columns = categoricalColumns
    OpVectorMetadata(getOutputFeatureName, columns, inputFeaturesToHistory(inN, stageName))
  }

  def makeIDMapVectorizerModelArgs(aggregatedStats: Array[TextMapStats]): IDMapVectorizerModelArgs = {
    val maxCard = $(maxCardinality)
    val minSup = $(minSupport)
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = $(cleanText)
    val shouldTrackNulls = $(trackNulls)

    val allFeatureInfo = aggregatedStats.toSeq.map { textMapStats =>
      textMapStats.keyValueCounts.toSeq.map { case (k, textStats) =>
        val isCat = textStats.valueCounts.size <= maxCard
        val topVals = if (isCat) {
          textStats.valueCounts
            .filter { case (_, count) => count >= minSup }
            .toSeq.sortBy(v => -v._2 -> v._1)
            .take($(topK)).map(_._1).toArray
        } else Array.empty[String]
        DropIDFeatureInfo(key = k, isCategorical = isCat, topValues = topVals)
      }
    }

    IDMapVectorizerModelArgs(
      allFeatureInfo = allFeatureInfo,
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues,
      shouldTrackNulls = shouldTrackNulls
    )
  }

  def fitFn(dataset: Dataset[Seq[IDMap#Value]]): SequenceModel[IDMap, OPVector] = {
    require(!dataset.isEmpty, "Input dataset cannot be empty")

    val maxCard = $(maxCardinality)
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = $(cleanText)

    implicit val testStatsSG: Semigroup[TextMapStats] = TextMapStats.semiGroup(maxCard)
    val valueStats: Dataset[Array[TextMapStats]] = dataset.map(
      _.map(computeTextMapStats(_, shouldCleanKeys, shouldCleanValues)).toArray
    )
    val aggregatedStats: Array[TextMapStats] = valueStats.reduce(_ + _)

    val dropIDMapVectorizerModelArgs = makeIDMapVectorizerModelArgs(aggregatedStats)

    val vecMetadata = makeVectorMetadata(dropIDMapVectorizerModelArgs)
    setMetadata(vecMetadata.toMetadata)

    new IDMapVectorizerModel(args = dropIDMapVectorizerModelArgs, operationName = operationName, uid = uid)

  }
}



/**
 * Info about each feature within a text map
 *
 * @param key           name of a feature
 * @param isCategorical indicate whether a feature is categorical or not
 * @param topValues     most common values of a feature (only for categoricals)
 */
case class DropIDFeatureInfo(key: String, isCategorical: Boolean, topValues: Array[String]) extends JsonLike


/**
 * Arguments for [[IDMapVectorizerModel]]
 *
 * @param allFeatureInfo    info about each feature with each text map
 * @param shouldCleanKeys   should clean feature keys
 * @param shouldCleanValues should clean feature values
 * @param shouldTrackNulls  should track nulls
 */
case class IDMapVectorizerModelArgs
(
  allFeatureInfo: Seq[Seq[DropIDFeatureInfo]],
  shouldCleanKeys: Boolean,
  shouldCleanValues: Boolean,
  shouldTrackNulls: Boolean
) extends JsonLike {
  val (categoricalFeatureInfo, textFeatureInfo) = allFeatureInfo.map { featureInfoSeq =>
    featureInfoSeq.partition {
      _.isCategorical
    }
  }.unzip
  val categoricalKeys = categoricalFeatureInfo.map(featureInfoSeq => featureInfoSeq.map(_.key))
  val textKeys = textFeatureInfo.map(featureInfoSeq => featureInfoSeq.map(_.key))
}


final class IDMapVectorizerModel
(
  val args: IDMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[IDMap]) extends SequenceModel[IDMap, OPVector](operationName = operationName, uid = uid)
  with TextTokenizerParams with TextMapPivotVectorizerModelFun[OPMap[String]] {

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

  def transformFn: Seq[IDMap] => OPVector = row => {
    val (rowCategorical, keysCategorical, rowText, keysText) = partitionRow(row)
    val categoricalVector = categoricalPivotFn(rowCategorical)

    categoricalVector
  }


}

