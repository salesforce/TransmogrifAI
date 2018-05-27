/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.insights

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.SparkModelConverter._
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.IntParam
import scala.collection.mutable.PriorityQueue

/**
 * Creates record level insights for model predictions. Takes the model to explain as a constructor argument.
 * The input feature is the feature vector fed into the model.
 * @param model         model instance that you wish to explain
 * @param uid           uid for instance
 */
class RecordInsightsLOCO[T <: SparkWrapperParams[_]]
(
  val model: T,
  uid: String = UID[RecordInsightsLOCO[_]]
) extends UnaryTransformer[OPVector, TextMap](operationName = "recordInsightsLOCO", uid = uid) {

  final val topK = new IntParam(
    parent = this, name = "topK",
    doc = "Number of insights to keep for each record"
  )
  def setTopK(value: Int): this.type = set(topK, value)
  def getTopK: Int = $(topK)
  setDefault(topK -> 20)

  private val modelApply = toOP(model.getSparkMlStage().map(_.asInstanceOf[Transformer])).transformFn
  private val labelDummy = RealNN(0.0)

  private lazy val featureInfo = OpVectorMetadata(getInputSchema()(in1.name)).getColumnHistory().map(_.toJson(false))

  override def transformFn: OPVector => TextMap = (features) => {
    val baseScore = modelApply(labelDummy, features).score
    val maxHeap = PriorityQueue.empty(MinScore)

    // TODO sparse implementation only works if changing values to zero - use dense vector to test effect of zeros
    val featuresSparse = features.value.toSparse
    val featureArray = featuresSparse.indices.zip(featuresSparse.values)
    val filledSize = featureArray.length
    val featureSize = featuresSparse.size

    val k = $(topK)
    var i = 0
    while (i < filledSize) {
      val (oldInd, oldVal) = featureArray(i)
      featureArray.update(i, (oldInd, 0))
      val score = modelApply(labelDummy, OPVector(Vectors.sparse(featureSize, featureArray))).score
      val diffs = baseScore.zip(score).map{ case (b, s) => b - s }
      val max = diffs.maxBy(math.abs)
      maxHeap.enqueue((i, max, diffs))
      if (i >= k) maxHeap.dequeue()
      featureArray.update(i, (oldInd, oldVal))
      i += 1
    }

    val top = maxHeap.dequeueAll
    top.map{ case (k, _, v) => RecordInsightsParser.insightToText(featureInfo(k), v) }
      .toMap.toTextMap
  }
}


private[insights] object MinScore extends Ordering[(Int, Double, Array[Double])] {
  def compare(x: (Int, Double, Array[Double]), y: (Int, Double, Array[Double])): Int =
    math.abs(y._2) compare math.abs(x._2)
}
