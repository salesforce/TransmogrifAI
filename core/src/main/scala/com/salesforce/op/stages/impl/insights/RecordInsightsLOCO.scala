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

package com.salesforce.op.stages.impl.insights

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.stages.impl.selector.{ProblemType, SelectedModel}
import com.salesforce.op.stages.sparkwrappers.specific.OpPredictorWrapperModel
import com.salesforce.op.stages.sparkwrappers.specific.SparkModelConverter._
import com.salesforce.op.utils.spark.OpVectorMetadata
import enumeratum.{Enum, EnumEntry}
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{IntParam, Param}

import scala.collection.mutable.PriorityQueue

/**
 * Creates record level insights for model predictions. Takes the model to explain as a constructor argument.
 * The input feature is the feature vector fed into the model.
 *
 * The map's contents are different regarding the value of the topKStrategy param (only for Binary Classification
 * and Regression) :
 * - If PositiveNegative, returns at most 2 * topK elements : the topK most positive and the topK most negative
 * derived features based on the LOCO insight.
 * - If Abs, returns at most topK elements : the topK derived features having highest absolute value of LOCO score.
 * @param model model instance that you wish to explain
 * @param uid   uid for instance
 */
@Experimental
class RecordInsightsLOCO[T <: Model[T]]
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

  final val topKStrategy = new Param[String](parent = this, name = "topKStrategy",
    doc = "Whether returning topK based on absolute value or topK positives and negatives. Only for Binary " +
      "Classification and Regression."
  )

  def setTopKStrategy(strategy: TopKStrategy): this.type = set(topKStrategy, strategy.entryName)

  def getTopKStrategy: TopKStrategy = TopKStrategy.withName($(topKStrategy))

  setDefault(topKStrategy, TopKStrategy.Abs.entryName)

  private val modelApply = model match {
    case m: SelectedModel => m.transformFn
    case m: OpPredictorWrapperModel[_] => m.transformFn
    case m => toOPUnchecked(m).transformFn
  }
  private val labelDummy = RealNN(0.0)

  private lazy val featureInfo = OpVectorMetadata(getInputSchema()(in1.name)).getColumnHistory().map(_.toJson(false))

  private lazy val vectorDummy = Array.fill(featureInfo.length)(0.0).toOPVector

  private[insights] lazy val problemType = modelApply(labelDummy, vectorDummy).score.length match {
    case 0 => ProblemType.Unknown
    case 1 => ProblemType.Regression
    case 2 => ProblemType.BinaryClassification
    case n if (n > 2) => {
      log.info("MultiClassification Problem : Top K LOCOs by absolute value")
      ProblemType.MultiClassification
    }
  }

  private def computeDiffs(i: Int, oldInd: Int, featureArray: Array[(Int, Double)], featureSize: Int,
    baseScore: Array[Double]): Array[Double] = {
    featureArray.update(i, (oldInd, 0))
    val score = modelApply(labelDummy, OPVector(Vectors.sparse(featureSize, featureArray))).score
    val diffs = baseScore.zip(score).map { case (b, s) => b - s }
    diffs
  }

  private def returnTopAbs(filledSize: Int, featureArray: Array[(Int, Double)], featureSize: Int,
    baseScore: Array[Double], k: Int): Seq[(Int, Double, Array[Double])] = {
    var i = 0
    val absoluteMaxHeap = PriorityQueue.empty(AbsScore)
    var count = 0
    while (i < filledSize) {
      val (oldInd, oldVal) = featureArray(i)
      featureArray.update(i, (oldInd, 0))
      val score = modelApply(labelDummy, OPVector(Vectors.sparse(featureSize, featureArray))).score
      val diffs = baseScore.zip(score).map { case (b, s) => b - s }
      val max = diffs.maxBy(math.abs)
      if (max != 0) { // Not keeping LOCOs with value 0.0
        absoluteMaxHeap.enqueue((i, max, diffs))
        count += 1
        if (count > k) absoluteMaxHeap.dequeue()
      }
      featureArray.update(i, (oldInd, oldVal))
      i += 1
    }

    val topAbs = absoluteMaxHeap.dequeueAll


    topAbs.sortBy { case (_, v, _) => -math.abs(v) }

  }

  private def returnTopPosNeg(filledSize: Int, featureArray: Array[(Int, Double)], featureSize: Int,
    baseScore: Array[Double], k: Int): Seq[(Int, Double, Array[Double])] = {
    // Heap that will contain the top K positive LOCO values
    val positiveMaxHeap = PriorityQueue.empty(MinScore)
    // Heap that will contain the top K negative LOCO values
    val negativeMaxHeap = PriorityQueue.empty(MaxScore)
    // for each element of the feature vector != 0.0
    // Size of positive heap
    var positiveCount = 0
    // Size of negative heap
    var negativeCount = 0
    for {i <- 0 until filledSize} {
      val (oldInd, oldVal) = featureArray(i)
      val diffs = computeDiffs(i, oldInd, featureArray, featureSize, baseScore)
      val max = if (problemType == ProblemType.Regression) diffs(0) else diffs(1)

      if (max > 0.0) { // if positive LOCO then add it to positive heap
        positiveMaxHeap.enqueue((i, max, diffs))
        positiveCount += 1
        if (positiveCount > k) { // remove the lowest element if the heap size goes from 5 to 6
          positiveMaxHeap.dequeue()
        }
      } else if (max < 0.0) { // if negative LOCO then add it to negative heap
        negativeMaxHeap.enqueue((i, max, diffs))
        negativeCount += 1
        if (negativeCount > k) { // remove the highest element if the heap size goes from 5 to 6
          negativeMaxHeap.dequeue()
        } // Not keeping LOCOs with value 0
      }
      featureArray.update(i, (oldInd, oldVal))
    }
    val topPositive = positiveMaxHeap.dequeueAll
    val topNegative = negativeMaxHeap.dequeueAll
    (topPositive ++ topNegative).sortBy { case (_, v, _) => -v }
  }

  override def transformFn: OPVector => TextMap = (features) => {
    val baseScore = modelApply(labelDummy, features).score

    // TODO sparse implementation only works if changing values to zero - use dense vector to test effect of zeros
    val featuresSparse = features.value.toSparse
    val featureArray = featuresSparse.indices.zip(featuresSparse.values)
    val filledSize = featureArray.length
    val featureSize = featuresSparse.size

    val k = $(topK)
    val top = getTopKStrategy match {
      case s if s == TopKStrategy.Abs || problemType == ProblemType.MultiClassification =>  returnTopAbs(filledSize,
        featureArray, featureSize, baseScore, k)
      case s if s == TopKStrategy.PositiveNegative => returnTopPosNeg(filledSize, featureArray, featureSize, baseScore,
        k)
    }


    top.map { case (k, _, v) => RecordInsightsParser.insightToText(featureInfo(featureArray(k)._1), v) }
      .toMap.toTextMap
  }
}


private[insights] object AbsScore extends Ordering[(Int, Double, Array[Double])] {
  def compare(x: (Int, Double, Array[Double]), y: (Int, Double, Array[Double])): Int =
    math.abs(y._2) compare math.abs(x._2)
}

/**
 * Ordering of the heap that removes lowest score
 */
private object MinScore extends Ordering[(Int, Double, Array[Double])] {
  def compare(x: (Int, Double, Array[Double]), y: (Int, Double, Array[Double])): Int =
    y._2 compare x._2
}

/**
 * Ordering of the heap that removes highest score
 */
private object MaxScore extends Ordering[(Int, Double, Array[Double])] {
  def compare(x: (Int, Double, Array[Double]), y: (Int, Double, Array[Double])): Int =
    x._2 compare y._2
}

sealed abstract class TopKStrategy(val name: String) extends EnumEntry with Serializable

object TopKStrategy extends Enum[TopKStrategy] {
  val values = findValues

  case object Abs extends TopKStrategy("abs")

  case object PositiveNegative extends TopKStrategy("positive and negative")

}
