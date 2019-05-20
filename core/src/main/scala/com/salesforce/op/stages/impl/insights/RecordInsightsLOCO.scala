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
import com.salesforce.op.stages.impl.feature.{SmartTextMapVectorizer, SmartTextVectorizer}
import com.salesforce.op.stages.impl.selector.SelectedModel
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
  private lazy val vectorMetadata = OpVectorMetadata(getInputSchema()(in1.name))
  private lazy val histories = vectorMetadata.getColumnHistory()
  private lazy val featureInfo = histories.map(_.toJson(false))

  /**
   * These are the name of the stages we want to perform an aggregation of the LOCO results over ferived features
   */
  private val smartTextClassName = classOf[SmartTextVectorizer[_]].getSimpleName
  private val smartTextMapClassName = classOf[SmartTextMapVectorizer[_]].getSimpleName
  // Indices of features derived from SmartText(Map)Vectorizer
  private lazy val textIndices = histories.filter(_.parentFeatureStages.exists(s => s.contains(smartTextClassName)
    || s.contains(smartTextMapClassName)))
    .map(_.index)


  private def computeDiffs(i: Int, oldInd: Int, oldVal: Double, featureArray: Array[(Int, Double)], featureSize: Int,
    baseScore: Array[Double]): Array[Double] = {
    featureArray.update(i, (oldInd, 0))
    val score = modelApply(labelDummy, OPVector(Vectors.sparse(featureSize, featureArray))).score
    val diffs = baseScore.zip(score).map { case (b, s) => b - s }
    featureArray.update(i, (oldInd, oldVal))
    diffs
  }

  private def sumArray(left: Array[Double], right: Array[Double]): Array[Double] = {
    left.zipAll(right, 0.0, 0.0).map { case (l, r) => l + r }
  }


  private def returnTopPosNeg(filledSize: Int, featureArray: Array[(Int, Double)], featureSize: Int,
    baseScore: Array[Double], k: Int, indexToExamine: Int): Seq[(Int, Double, Array[Double])] = {
    // Heap that will contain the top K positive LOCO values
    val positiveMaxHeap = PriorityQueue.empty(MinScore)
    // Heap that will contain the top K negative LOCO values
    val negativeMaxHeap = PriorityQueue.empty(MaxScore)
    // for each element of the feature vector != 0.0
    // Size of positive heap
    var positiveCount = 0
    // Size of negative heap
    var negativeCount = 0

    def addToHeaps(i: Int, max: Double, diffToExamine: Array[Double]): Unit = {

      if (max > 0.0) { // if positive LOCO then add it to positive heap
        positiveMaxHeap.enqueue((i, max, diffToExamine))
        positiveCount += 1
        if (positiveCount > k) { // remove the lowest element if the heap size goes from 5 to 6
          positiveMaxHeap.dequeue()
        }
      } else if (max < 0.0) { // if negative LOCO then add it to negative heap
        negativeMaxHeap.enqueue((i, max, diffToExamine))
        negativeCount += 1
        if (negativeCount > k) { // remove the highest element if the heap size goes from 5 to 6
          negativeMaxHeap.dequeue()
        } // Not keeping LOCOs with value 0
      }
    }

    var i = 0
    val aggregationMap = scala.collection.mutable.Map.empty[String, (Array[Int], Array[Double])]
    while (i < filledSize) {
      val (oldInd, oldVal) = featureArray(i)
      val diffToExamine = computeDiffs(i, oldInd, oldVal, featureArray, featureSize, baseScore)
      val max = diffToExamine(indexToExamine)

      // Let's check the indicator value and descriptor value
      val history = histories(oldInd)
      val parentStages = history.parentFeatureStages
      val indValue = history.indicatorValue
      val descValue = history.descriptorValue

      // If those values are empty, the field is likely to be a derived text feature (hashing tf output)
      if (indValue.isEmpty && descValue.isEmpty) {
        // Name of the field
        val rawName = parentStages match {
          case s if s.exists(_.contains(smartTextClassName)) => Some(history.parentFeatureOrigins.head)
          case s if s.exists(_.contains(smartTextMapClassName)) => history.grouping
          case _ => None
        }
        rawName.map { name =>
          // Update the aggregation map
          val (indices, array) = aggregationMap.getOrElse(name, (Array.empty[Int], Array.empty[Double]))
          aggregationMap(name) = (indices :+ i, sumArray(array, diffToExamine))
        }
      } else {
        addToHeaps(i, max, diffToExamine)
      }
      i += 1
    }
    // Adding LOCO results from aggregation map into heaps
    aggregationMap.foreach {
      case (_, (indices, ar)) => {
        val i = indices.head // The index here is arbitrary
        val n = indices.length
        val diffToExamine = ar.map(_ / n)
        val max = diffToExamine(indexToExamine)
        addToHeaps(i, max, diffToExamine)
      }
    }

    val topPositive = positiveMaxHeap.dequeueAll
    val topNegative = negativeMaxHeap.dequeueAll
    topPositive ++ topNegative
  }

  override def transformFn: OPVector => TextMap = features => {

    val baseResult = modelApply(labelDummy, features)
    val baseScore = baseResult.score

    // TODO sparse implementation only works if changing values to zero - use dense vector to test effect of zeros
    val featuresSparse = features.value.toSparse
    // Besides non 0 values, we want to check the text features as well
    val indices = (featuresSparse.indices ++ textIndices).toSet
    val filledSize = indices.size
    val featureArray = indices.map(i => i -> features.value(i)).toArray
    val featureSize = featuresSparse.size

    val k = $(topK)
    // Index where to examine the difference in the prediction vector
    val indexToExamine = baseScore.length match {
      case 0 => throw new RuntimeException("model does not produce scores for insights")
      case 1 => 0
      case 2 => 1
      case n if n > 2 => baseResult.prediction.toInt
    }
    val topPosNeg = returnTopPosNeg(filledSize, featureArray, featureSize, baseScore, k, indexToExamine)
    val top = getTopKStrategy match {
      case TopKStrategy.Abs => topPosNeg.sortBy { case (_, v, _) => -math.abs(v) }.take(k)
      case TopKStrategy.PositiveNegative => topPosNeg.sortBy { case (_, v, _) => -v }
    }

    top.map { case (k, _, v) => RecordInsightsParser.insightToText(featureInfo(featureArray(k)._1), v) }
      .toMap.toTextMap
  }
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
