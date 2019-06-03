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
import com.salesforce.op.stages.impl.selector.SelectedModel
import com.salesforce.op.stages.sparkwrappers.specific.OpPredictorWrapperModel
import com.salesforce.op.stages.sparkwrappers.specific.SparkModelConverter._
import com.salesforce.op.utils.spark.OpVectorMetadata
import enumeratum.{Enum, EnumEntry}
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{IntParam, Param}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Creates record level insights for model predictions. Takes the model to explain as a constructor argument.
 * The input feature is the feature vector fed into the model.
 *
 * The map's contents are different regarding the value of the topKStrategy param (only for Binary Classification
 * and Regression) :
 * - If PositiveNegative, returns at most 2 * topK elements : the topK most positive and the topK most negative
 * derived features based on the LOCO insight.For MultiClassification, the value is from the predicted class
 * (i.e. the class having the highest probability)
 * - If Abs, returns at most topK elements : the topK derived features having highest absolute value of LOCO score.
 *
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
    doc = "Whether returning topK based on absolute value or topK positives and negatives. For MultiClassification," +
      " the value is from the predicted class (i.e. the class having the highest probability)"
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
  private lazy val histories = OpVectorMetadata(getInputSchema()(in1.name)).getColumnHistory()
  private lazy val featureInfo = histories.map(_.toJson(false))

  /**
   * These are the name of the types we want to perform an aggregation of the LOCO results over derived features
   */
  private val textTypes =
    Set(FeatureType.typeName[Text], FeatureType.typeName[TextArea], FeatureType.typeName[TextList])
  private val textMapTypes =
    Set(FeatureType.typeName[TextMap], FeatureType.typeName[TextAreaMap])

  // Indices of features derived from Text(Map)Vectorizer
  private lazy val textFeatureIndices = histories
    .filter(_.parentFeatureType.exists((textTypes ++ textMapTypes).contains))
    .map(_.index)
    .distinct.sorted

  private def computeDiffs
  (
    i: Int,
    oldInd: Int,
    oldVal: Double,
    featureArray: Array[(Int, Double)],
    featureSize: Int,
    baseScore: Array[Double]
  ): Array[Double] = {
    featureArray.update(i, (oldInd, 0.0))
    val score = modelApply(labelDummy, Vectors.sparse(featureSize, featureArray).toOPVector).score
    val diffs = baseScore.zip(score).map { case (b, s) => b - s }
    featureArray.update(i, (oldInd, oldVal))
    diffs
  }

  private def sumArrays(left: Array[Double], right: Array[Double]): Array[Double] = {
    left.zipAll(right, 0.0, 0.0).map { case (l, r) => l + r }
  }

  private def returnTopPosNeg
  (
    featureArray: Array[(Int, Double)],
    featureSize: Int,
    baseScore: Array[Double],
    k: Int,
    indexToExamine: Int
  ): Seq[LOCOValue] = {

    val minMaxHeap = new MinMaxHeap(k)
    val aggregationMap = mutable.Map.empty[String, (Array[Int], Array[Double])]
    for {i <- featureArray.indices} {
      val (oldInd, oldVal) = featureArray(i)
      val diffToExamine = computeDiffs(i, oldInd, oldVal, featureArray, featureSize, baseScore)
      val history = histories(oldInd)

      // Let's check the indicator value and descriptor value
      // If those values are empty, the field is likely to be a derived text feature (hashing tf output)
      if (textFeatureIndices.contains(oldInd) && history.indicatorValue.isEmpty && history.descriptorValue.isEmpty) {
        // Name of the field
        val rawName = history.parentFeatureType match {
          case s if s.exists(textMapTypes.contains) => history.grouping
          case s if s.exists(textTypes.contains) => history.parentFeatureOrigins.headOption
          case s => throw new Error(s"type should be Text or TextMap, here ${s.mkString(",")}")
        }
        // Update the aggregation map
        for {name <- rawName} {
          val key = name + "_" + history.parentFeatureStages.mkString(",")
          val (indices, array) = aggregationMap.getOrElse(key, (Array.empty[Int], Array.empty[Double]))
          aggregationMap.update(key, (indices :+ i, sumArrays(array, diffToExamine)))
        }
      } else {
        minMaxHeap enqueue LOCOValue(i, diffToExamine(indexToExamine), diffToExamine)
      }
    }

    // Adding LOCO results from aggregation map into heaps
    for {(indices, ar) <- aggregationMap.values} {
      // The index here is arbitrary
      val (i, n) = (indices.head, indices.length)
      val diffToExamine = ar.map(_ / n)
      minMaxHeap enqueue LOCOValue(i, diffToExamine(indexToExamine), diffToExamine)
    }

    minMaxHeap.dequeueAll
  }

  override def transformFn: OPVector => TextMap = features => {
    val baseResult = modelApply(labelDummy, features)
    val baseScore = baseResult.score

    // TODO: sparse implementation only works if changing values to zero - use dense vector to test effect of zeros
    val featuresSparse = features.value.toSparse
    val res = ArrayBuffer.empty[(Int, Double)]
    featuresSparse.foreachActive((i, v) => res += i -> v)
    // Besides non 0 values, we want to check the text features as well
    textFeatureIndices.foreach(i => if (!featuresSparse.indices.contains(i)) res += i -> 0.0)
    val featureArray = res.toArray
    val featureSize = featuresSparse.size

    val k = $(topK)
    // Index where to examine the difference in the prediction vector
    val indexToExamine = baseScore.length match {
      case 0 => throw new RuntimeException("model does not produce scores for insights")
      case 1 => 0
      case 2 => 1
      // For MultiClassification, the value is from the predicted class(i.e. the class having the highest probability)
      case n if n > 2 => baseResult.prediction.toInt
    }
    val topPosNeg = returnTopPosNeg(featureArray, featureSize, baseScore, k, indexToExamine)
    val top = getTopKStrategy match {
      case TopKStrategy.Abs => topPosNeg.sortBy { case LOCOValue(_, v, _) => -math.abs(v) }.take(k)
      // Take top K positive and top K negative LOCOs, hence 2 * K
      case TopKStrategy.PositiveNegative => topPosNeg.sortBy { case LOCOValue(_, v, _) => -v }.take(2 * k)
    }

    top.map { case LOCOValue(i, _, diffs) =>
      RecordInsightsParser.insightToText(featureInfo(featureArray(i)._1), diffs)
    }.toMap.toTextMap
  }

}

/**
 * Heap to keep top K of min and max LOCO values
 *
 * @param k number of values to keep
 */
private class MinMaxHeap(k: Int) {

  // Heap that will contain the top K positive LOCO values
  private val positives = mutable.PriorityQueue.empty(MinScore)

  // Heap that will contain the top K negative LOCO values
  private val negatives = mutable.PriorityQueue.empty(MaxScore)

  def enqueue(loco: LOCOValue): Unit = {
    // Not keeping LOCOs with value 0, i.e. for each element of the feature vector != 0.0
    if (loco.value > 0.0) { // if positive LOCO then add it to positive heap
      positives.enqueue(loco)
      // remove the lowest element if the heap size goes above k
      if (positives.length > k) positives.dequeue()
    } else if (loco.value < 0.0) { // if negative LOCO then add it to negative heap
      negatives.enqueue(loco)
      // remove the highest element if the heap size goes above k
      if (negatives.length > k) negatives.dequeue()
    }
  }

  def dequeueAll: Seq[LOCOValue] = positives.dequeueAll ++ negatives.dequeueAll

}

/**
 * LOCO value container
 *
 * @param i     feature value index
 * @param value value - min or max, depending on the ordering
 * @param diffs scores diff
 */
private case class LOCOValue(i: Int, value: Double, diffs: Array[Double])

/**
 * Ordering of the heap that removes lowest score
 */
private object MinScore extends Ordering[LOCOValue] {
  def compare(x: LOCOValue, y: LOCOValue): Int = y.value compare x.value
}

/**
 * Ordering of the heap that removes highest score
 */
private object MaxScore extends Ordering[LOCOValue] {
  def compare(x: LOCOValue, y: LOCOValue): Int = x.value compare y.value
}

sealed abstract class TopKStrategy(val name: String) extends EnumEntry with Serializable

object TopKStrategy extends Enum[TopKStrategy] {
  val values = findValues
  case object Abs extends TopKStrategy("abs")
  case object PositiveNegative extends TopKStrategy("positive and negative")
}
