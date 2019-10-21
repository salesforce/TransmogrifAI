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
import com.salesforce.op.stages.impl.feature.TimePeriod
import com.salesforce.op.stages.impl.selector.SelectedModel
import com.salesforce.op.stages.sparkwrappers.specific.OpPredictorWrapperModel
import com.salesforce.op.stages.sparkwrappers.specific.SparkModelConverter._
import com.salesforce.op.utils.spark.RichVector.RichSparseVector
import com.salesforce.op.utils.spark.{OpVectorColumnHistory, OpVectorMetadata}
import enumeratum.{Enum, EnumEntry}
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.param.{IntParam, Param, Params}

import scala.collection.mutable


trait RecordInsightsLOCOParams extends Params {

  final val topK = new IntParam(
    parent = this, name = "topK",
    doc = "Number of insights to keep for each record"
  )
  def setTopK(value: Int): this.type = set(topK, value)
  def getTopK: Int = $(topK)

  final val topKStrategy = new Param[String](parent = this, name = "topKStrategy",
    doc = "Whether returning topK based on absolute value or topK positives and negatives. For MultiClassification," +
      " the value is from the predicted class (i.e. the class having the highest probability)"
  )
  def setTopKStrategy(strategy: TopKStrategy): this.type = set(topKStrategy, strategy.entryName)
  def getTopKStrategy: TopKStrategy = TopKStrategy.withName($(topKStrategy))

  final val vectorAggregationStrategy = new Param[String](parent = this, name = "vectorAggregationStrategy",
    doc = "Aggregate text/date vector by " +
      "1. LeaveOutVector strategy - calculate the loco by leaving out the entire vector or" +
      "2. Avg strategy - calculate the loco for each column of the vector and then average all the locos."
  )
  def setVectorAggregationStrategy(strategy: VectorAggregationStrategy): this.type =
    set(vectorAggregationStrategy, strategy.entryName)
  def getVectorAggregationStrategy: VectorAggregationStrategy = VectorAggregationStrategy.withName(
    $(vectorAggregationStrategy))


  setDefault(
    topK -> 20,
    topKStrategy -> TopKStrategy.Abs.entryName,
    vectorAggregationStrategy -> VectorAggregationStrategy.Avg.entryName
  )
}

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
) extends UnaryTransformer[OPVector, TextMap](operationName = "recordInsightsLOCO", uid = uid)
  with RecordInsightsLOCOParams {

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
  private val dateTypes =
    Set(FeatureType.typeName[Date], FeatureType.typeName[DateTime])
  private val dateMapTypes =
    Set(FeatureType.typeName[DateMap], FeatureType.typeName[DateTimeMap])

  // Map[FeatureName(Date/Text), VectorSize]
  private lazy val aggFeaturesSize: Map[String, Int] = getFeaturesSize(isTextIndex) ++ getFeaturesSize(isDateIndex)

  private def getFeaturesSize(predicate: OpVectorColumnHistory => Boolean): Map[String, Int] = histories
    .filter(predicate)
    .groupBy { h => getRawFeatureName(h).get }
    .mapValues(_.length).view.toMap

  private def isTextIndex(h: OpVectorColumnHistory): Boolean = {
    h.parentFeatureType.exists((textTypes ++ textMapTypes).contains) &&
      h.indicatorValue.isEmpty && h.descriptorValue.isEmpty
  }

  private def isDateIndex(h: OpVectorColumnHistory): Boolean = {
    h.parentFeatureType.exists((dateTypes ++ dateMapTypes).contains) && h.descriptorValue.isDefined
  }

  private def computeDiff
  (
    featureSparse: SparseVector,
    baseScore: Array[Double]
  ): Array[Double] = {
    val score = modelApply(labelDummy, featureSparse.toOPVector).score
    (baseScore, score).zipped.map { case (b, s) => b - s }
  }

  private def sumArrays(left: Array[Double], right: Array[Double]): Array[Double] = {
    left.zipAll(right, 0.0, 0.0).map { case (l, r) => l + r }
  }

  /**
   * Optionally convert columnMetadata's descriptorValue like
   * "y_DayOfWeek", "x_DayOfWeek" to TimePeriod enum - DayOfWeek.
   * @return Option[TimePeriod]
   */
  private def convertToTimePeriod(descriptorValue: String): Option[TimePeriod] =
    descriptorValue.split("_").lastOption.flatMap(TimePeriod.withNameInsensitiveOption)

  private def getRawFeatureName(history: OpVectorColumnHistory): Option[String] = {
    val groupSuffix = history.grouping.map("_" + _).getOrElse("")
    val name = history.parentFeatureOrigins.headOption.map(_ + groupSuffix)

    // If the descriptor value of a derived feature exists, then we check if it is
    // from unit circle transformer. We aggregate such features for each (rawFeatureName, timePeriod).
    // TODO : Filter by parentStage (DateToUnitCircleTransformer & DateToUnitCircleVectorizer) once the bug in the
    //  feature history after multiple transformations has been fixed
    name.map { n =>
      val timePeriodName = if ((dateTypes ++ dateMapTypes).exists(history.parentFeatureType.contains)) {
        history.descriptorValue
          .flatMap(convertToTimePeriod)
          .map(p => "_" + p.entryName)
      } else None
      n + timePeriodName.getOrElse("")
    }
  }

  private def aggregateDiffs(
    featureSparse: SparseVector,
    aggIndices: Array[(Int, Int)],
    strategy: VectorAggregationStrategy,
    baseScore: Array[Double],
    featureSize: Int
  ): Array[Double] = {
    strategy match {
      case VectorAggregationStrategy.Avg =>
        aggIndices
          .map { case (i, oldInd) => computeDiff(featureSparse.copy.updated(i, oldInd, 0.0), baseScore) }
          .foldLeft(Array.empty[Double])(sumArrays)
          .map( _ / featureSize)

      case VectorAggregationStrategy.LeaveOutVector =>
        val copyFeatureSparse = featureSparse.copy
        aggIndices.foreach {case (i, oldInd) => copyFeatureSparse.updated(i, oldInd, 0.0)}
        computeDiff(copyFeatureSparse, baseScore)
    }
  }

  private def returnTopPosNeg
  (
    featureSparse: SparseVector,
    baseScore: Array[Double],
    k: Int,
    indexToExamine: Int
  ): Seq[LOCOValue] = {
    val minMaxHeap = new MinMaxHeap(k)

    // Map[FeatureName, (Array[SparseVectorIndices], Array[ActualIndices])
    val aggActiveIndices = mutable.Map.empty[String, Array[(Int, Int)]]

    (0 until featureSparse.size, featureSparse.indices).zipped.foreach { case (i: Int, oldInd: Int) =>
      val history = histories(oldInd)
      history match {
        case h if isTextIndex(h) || isDateIndex(h) => {
          for {name <- getRawFeatureName(h)} {
            val indices = aggActiveIndices.getOrElse(name, (Array.empty[(Int, Int)]))
            aggActiveIndices.update(name, indices :+ (i, oldInd))
          }
        }
        case _ => {
          val diffToExamine = computeDiff(featureSparse.copy.updated(i, oldInd, 0.0), baseScore)
          minMaxHeap enqueue LOCOValue(i, diffToExamine(indexToExamine), diffToExamine)
        }
      }
    }

    // Aggregate active indices of each text feature and date feature based on vector aggregate strategy.
    aggActiveIndices.foreach {
      case (name, aggIndices) =>
        val diffToExamine = aggregateDiffs(featureSparse, aggIndices,
          getVectorAggregationStrategy, baseScore, aggFeaturesSize.get(name).get)
        minMaxHeap enqueue LOCOValue(aggIndices.head._1, diffToExamine(indexToExamine), diffToExamine)
    }

    minMaxHeap.dequeueAll
  }

  override def transformFn: OPVector => TextMap = features => {
    val baseResult = modelApply(labelDummy, features)
    val baseScore = baseResult.score

    // TODO: sparse implementation only works if changing values to zero - use dense vector to test effect of zeros
    val featuresSparse = features.value.toSparse

    val k = $(topK)
    // Index where to examine the difference in the prediction vector
    val indexToExamine = baseScore.length match {
      case 0 => throw new RuntimeException("model does not produce scores for insights")
      case 1 => 0
      case 2 => 1
      // For MultiClassification, the value is from the predicted class(i.e. the class having the highest probability)
      case n if n > 2 => baseResult.prediction.toInt
    }
    val topPosNeg = returnTopPosNeg(featuresSparse, baseScore, k, indexToExamine)
    val top = getTopKStrategy match {
      case TopKStrategy.Abs => topPosNeg.sortBy { case LOCOValue(_, v, _) => -math.abs(v) }.take(k)
      // Take top K positive and top K negative LOCOs, hence 2 * K
      case TopKStrategy.PositiveNegative => topPosNeg.sortBy { case LOCOValue(_, v, _) => -v }.take(2 * k)
    }

    val allIndices = featuresSparse.indices
    top.map { case LOCOValue(i, _, diffs) =>
      RecordInsightsParser.insightToText(featureInfo(allIndices(i)), diffs)
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


sealed abstract class VectorAggregationStrategy(val name: String) extends EnumEntry with Serializable

object VectorAggregationStrategy extends Enum[VectorAggregationStrategy] {
  val values = findValues
  case object LeaveOutVector extends
    VectorAggregationStrategy("calculate the loco by leaving out the entire vector")
  case object Avg extends
    VectorAggregationStrategy("calculate the loco for each column of the vector and then average all the locos")
}
