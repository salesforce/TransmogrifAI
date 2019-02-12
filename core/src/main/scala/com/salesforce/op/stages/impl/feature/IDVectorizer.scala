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
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.Semigroup
import org.apache.spark.sql.{Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import com.salesforce.op.utils.spark.RichDataset._


/**
 * Convert a sequence of id features into a vector by detecting categoricals.
 * A categorical will be represented as a vector consisting of occurrences of top K most common values of that feature
 * plus occurrences of non top k values and a null indicator (if enabled).
 * Non-categoricals will be removed.
 *
 * @param uid uid for instance
 */
class IDVectorizer
(uid: String = UID[IDVectorizer])
  extends SequenceEstimator[ID, OPVector](operationName = "idvec", uid = uid)
    with PivotParams with CleanTextFun with SaveOthersParams
    with TrackNullsParam with MinSupportParam with MaxCardinalityParams with OneHotFun {

  private implicit val textStatsSeqEnc: Encoder[Array[TextStats]] = ExpressionEncoder[Array[TextStats]]()


  def fitFn(dataset: Dataset[Seq[ID#Value]]): SequenceModel[ID, OPVector] = {
    require(!dataset.isEmpty, "Input dataset cannot be empty")

    val maxCard = $(maxCardinality)
    val shouldCleanText = $(cleanText)

    implicit val testStatsSG: Semigroup[TextStats] = TextStats.semiGroup(maxCard)
    val valueStats: Dataset[Array[TextStats]] = dataset.map(_.map(computeTextStats).toArray)
    val aggregatedStats: Array[TextStats] = valueStats.reduce(_ + _)

    val (isCategorical, topValues) = aggregatedStats.map { stats =>
      val isCategorical = stats.valueCounts.size <= maxCard
      val topValues = stats.valueCounts
        .filter { case (_, count) => count >= $(minSupport) }
        .toSeq.sortBy(v => -v._2 -> v._1)
        .take($(topK)).map(_._1)
      isCategorical -> topValues
    }.unzip

    val idParams = IDVectorizerModelArgs(
      isCategorical = isCategorical,
      topValues = topValues,
      shouldTrackNulls = $(trackNulls)
    )

    val vecMetadata = makeVectorMetadata(idParams)
    setMetadata(vecMetadata.toMetadata)

    new IDVectorizerModel(args = idParams, operationName = operationName, uid = uid)

  }

  private def computeTextStats(text: ID#Value): TextStats = {
    val valueCounts = text match {
      case Some(v) => Map(cleanTextFn(v, false) -> 1)
      case None => Map.empty[String, Int]
    }
    TextStats(valueCounts)
  }

  private def inputFeaturesToHistory(tf: Array[TransientFeature], thisStageName: String): Map[String, FeatureHistory] =
    tf.map(f => f.name -> FeatureHistory(originFeatures = f.originFeatures, stages = f.stages :+ thisStageName)).toMap

  private def makeVectorMetadata(idParams: IDVectorizerModelArgs): OpVectorMetadata = {
    require(inN.length == idParams.isCategorical.length)

    val (categoricalFeatures, textFeatures) =
      CategoricalDetection.partition[TransientFeature](inN, idParams.isCategorical)

    // build metadata describing output
    val shouldTrackNulls = $(trackNulls)
    val unseen = Option($(unseenName))

    val categoricalColumns = if (categoricalFeatures.nonEmpty) {
      makeVectorColumnMetadata(shouldTrackNulls, unseen, idParams.categoricalTopValues, categoricalFeatures)
    } else Array.empty[OpVectorColumnMetadata]

    val columns = categoricalColumns
    OpVectorMetadata(getOutputFeatureName, columns, inputFeaturesToHistory(inN, stageName))
  }
}


/**
 * Arguments for [[IDVectorizerModel]]
 *
 * @param isCategorical    is feature a categorical or not
 * @param topValues        top values to each feature
 * @param shouldTrackNulls should track nulls
 */
case class IDVectorizerModelArgs
(
  isCategorical: Array[Boolean],
  topValues: Array[Seq[String]],
  shouldTrackNulls: Boolean
) extends JsonLike {
  def categoricalTopValues: Array[Seq[String]] =
    topValues.zip(isCategorical).collect { case (top, true) => top }
}

final class IDVectorizerModel
(
  val args: IDVectorizerModelArgs,
  operationName: String,
  uid: String
) extends SequenceModel[ID, OPVector](operationName = operationName, uid = uid)
  with OneHotModelFun[ID] {

  override protected def convertToSet(in: ID): Set[String] = in.value.toSet

  def transformFn: Seq[ID] => OPVector = {
    val categoricalPivotFn: Seq[ID] => OPVector = pivotFn(
      topValues = args.categoricalTopValues,
      shouldCleanText = false,
      shouldTrackNulls = args.shouldTrackNulls
    )
    (row: Seq[ID]) => {
      val (rowCategorical, _) = CategoricalDetection.partition[ID](row.toArray, args.isCategorical)
      val categoricalVector: OPVector = categoricalPivotFn(rowCategorical)
      categoricalVector
    }
  }
}
