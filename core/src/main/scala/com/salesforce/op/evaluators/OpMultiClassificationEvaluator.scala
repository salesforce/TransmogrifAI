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

package com.salesforce.op.evaluators

import com.fasterxml.jackson.databind.annotation.JsonDeserialize
import com.salesforce.op.UID
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.Tuple2Semigroup
import com.salesforce.op.utils.spark.RichEvaluator._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{DoubleArrayParam, IntArrayParam}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}
import org.slf4j.LoggerFactory

/**
 * Instance to evaluate Multi Classification metrics
 * The metrics are  Precision, Recall, F1 and Error Rate
 * Default evaluation returns F1 score
 *
 * @param name           name of default metric
 * @param isLargerBetter is metric better if larger
 * @param uid            uid for instance
 */
private[op] class OpMultiClassificationEvaluator
(
  override val name: EvalMetric = OpEvaluatorNames.Multi,
  override val isLargerBetter: Boolean = true,
  override val uid: String = UID[OpMultiClassificationEvaluator]
) extends OpMultiClassificationEvaluatorBase[MultiClassificationMetrics](uid) {

  private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: MultiClassificationMetrics => Double = _.F1

  final val topNs = new IntArrayParam(
    parent = this,
    name = "topNs",
    doc = "sequence of topN values to use for threshold metrics",
    isValid = _.forall(_ > 0)
  )
  setDefault(topNs, Array(1, 3))

  def setTopNs(v: Array[Int]): this.type = set(topNs, v)

  final val thresholds = new DoubleArrayParam(
    parent = this,
    name = "thresholds",
    doc = "sequence of threshold values (must be in [0.0, 1.0]) to use for threshold metrics",
    isValid = _.forall(x => x >= 0.0 && x <= 1.0)
  )
  setDefault(thresholds, (0 to 100).map(_ / 100.0).toArray)

  def setThresholds(v: Array[Double]): this.type = set(thresholds, v)

  override def evaluateAll(data: Dataset[_]): MultiClassificationMetrics = {
    val labelColName = getLabelCol
    val dataUse = makeDataToUse(data, labelColName)

    val (predictionColName, rawPredictionColName, probabilityColName) = (getPredictionValueCol,
      getRawPredictionCol, getProbabilityCol)

    log.debug(
      "Evaluating metrics on columns :\n label : {}\n rawPrediction : {}\n prediction : {}\n probability : {}\n",
      labelColName, rawPredictionColName, predictionColName, probabilityColName
    )

    import dataUse.sparkSession.implicits._
    val rdd = dataUse.select(predictionColName, labelColName).as[(Double, Double)].rdd
    if (rdd.isEmpty()) {
      log.warn("The dataset is empty. Returning empty metrics.")
      MultiClassificationMetrics(0.0, 0.0, 0.0, 0.0,
        ThresholdMetrics(Seq.empty, Seq.empty, Map.empty, Map.empty, Map.empty))
    } else {
      val multiclassMetrics = new MulticlassMetrics(rdd)
      val error = 1.0 - multiclassMetrics.accuracy
      val precision = multiclassMetrics.weightedPrecision
      val recall = multiclassMetrics.weightedRecall
      val f1 = if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)

      val thresholdMetrics = calculateThresholdMetrics(
        data = dataUse.select(col(probabilityColName), col(labelColName).cast(DoubleType)).rdd.map{
          case Row(prob: Vector, label: Double) => (prob.toArray, label)
        },
        topNs = $(topNs),
        thresholds = $(thresholds)
      )

      val metrics = MultiClassificationMetrics(
        Precision = precision,
        Recall = recall,
        F1 = f1,
        Error = error,
        ThresholdMetrics = thresholdMetrics
      )

      log.info("Evaluated metrics: {}", metrics.toString)
      metrics
    }
  }


  /**
   * Function that calculates a set of threshold metrics for different topN values given an RDD of scores & labels,
   * a list of topN values to consider, and a list of thresholds to use.
   *
   * Output: ThresholdMetrics object, containing thresholds used, topN values used, and maps from topN value to
   * arrays of correct, incorrect, and no prediction counts at each threshold. Summing all three of these arrays
   * together should give an array where each entry the total number of rows in the input RDD.
   *
   * @param data       Input RDD consisting of (vector of score probabilities, label), where label corresponds to the
   *                   index of the true class and the score vector consists of probabilities for each class
   * @param topNs      Sequence of topN values to calculate threshold metrics for.
   *                   For example, if topN is Seq(1, 3, 10) then threshold metrics are calculated by considering if
   *                   the score of the true class is in the top 1, top 3, and top10 scores, respectively. If a topN
   *                   value is greater than the number of total classes,
   *                   then it will still be applied, but will have the same results as if that topN value = num classes
   * @param thresholds Sequence of threshold values applied to predicted probabilities, therefore they must be in the
   *                   range [0.0, 1.0]
   */
  def calculateThresholdMetrics(
    data: RDD[(Array[Double], Double)],
    topNs: Seq[Int],
    thresholds: Seq[Double]
  ): ThresholdMetrics = {
    require(thresholds.nonEmpty, "thresholds sequence in cannot be empty")
    require(thresholds.forall(x => x >= 0 && x <= 1.0), "thresholds sequence elements must be in the range [0, 1]")
    require(topNs.nonEmpty, "topN sequence in cannot be empty")
    require(topNs.forall(_ > 0), "topN sequence can only contain positive integers")

    type Label = Int
    type CorrIncorr = (Array[Long], Array[Long])
    type MetricsMap = Map[Label, CorrIncorr]

    val nThresholds = thresholds.length

    /**
     * Allocates an array of longs and fills it with a specified value from start until end
     */
    def arrayFill(size: Int)(start: Int, end: Int, value: Long) = {
      val res = new Array[Long](size)
      var i = start
      while (i < end) {
        res(i) = value
        i += 1
      }
      res
    }

    /**
     * First aggregation step turns an array of scores (as probabilities) and a single label (index of correct class)
     * into two arrays, correct and incorrect counts by threshold. Each array index corresponds to whether
     * the score counts as correct or incorrect at the threshold corresponding to that index.
     */
    def computeMetrics(scoresAndLabels: (Array[Double], Double)): MetricsMap = {
      val scores: Array[Double] = scoresAndLabels._1
      val label: Label = scoresAndLabels._2.toInt
      // The label may be unseen during model training, so treat scores for unseen classes as all being zero
      val trueClassScore: Double = if (scores.isDefinedAt(label)) scores(label) else 0.0
      val topNsAndScores: Map[Label, Array[(Double, Int)]] = topNs.map(t => t -> scores.zipWithIndex.sortBy(-_._1)
        .take(t)).toMap
      val topNScores: Map[Label, Array[Double]] = topNsAndScores.mapValues(_.map(_._1))
      // Doesn't matter which key you use since the scores are sorted
      val topScore: Double = topNScores.head._2.head
      val topNIndices: Map[Label, Array[Int]] = topNsAndScores.mapValues(_.map(_._2))

      // To calculate correct / incorrect counts per threshold, we just need to find the array index where the
      // true label score and the top score are no longer >= threshold.
      val trueScoreCutoffIndex: Int = {
        val idx = thresholds.indexWhere(_ > trueClassScore)
        if (idx < 0) nThresholds else idx
      }
      val maxScoreCutoffIndex: Int = {
        val idx = thresholds.indexWhere(_ > topScore)
        if (idx < 0) nThresholds else idx
      }
      topNs.view.map { t =>
        val correctCounts = if (topNIndices(t).contains(label)) {
          arrayFill(nThresholds)(start = 0, end = trueScoreCutoffIndex, value = 1L)
        } else new Array[Long](nThresholds)

        val incorrectCounts = if (topNIndices(t).contains(label)) {
          arrayFill(nThresholds)(start = trueScoreCutoffIndex, end = maxScoreCutoffIndex, value = 1L)
        } else arrayFill(nThresholds)(start = 0, end = maxScoreCutoffIndex, value = 1L)

        t -> (correctCounts, incorrectCounts)
      }.toMap[Label, CorrIncorr]
    }

    val zeroValue: MetricsMap =
      topNs
        .map(_ -> (new Array[Long](nThresholds), new Array[Long](nThresholds)))
        .toMap[Label, CorrIncorr]

    implicit val sgTuple2 = new Tuple2Semigroup[Array[Long], Array[Long]]()
    val agg: MetricsMap = data.treeAggregate[MetricsMap](zeroValue)(combOp = _ + _, seqOp = _ + computeMetrics(_))

    val nRows = data.count()
    ThresholdMetrics(
      topNs = topNs,
      thresholds = thresholds,
      correctCounts = agg.mapValues { case (cor, _) => cor.toSeq },
      incorrectCounts = agg.mapValues { case (_, incor) => incor.toSeq },
      noPredictionCounts = agg.mapValues { case (cor, incor) =>
        (Array.fill(nThresholds)(nRows) + cor.map(-_) + incor.map(-_)).toSeq
      }
    )
  }

  final protected def getMultiEvaluatorMetric(
    metricName: ClassificationEvalMetric,
    dataset: Dataset[_],
    default: => Double
  ): Double = {
    val labelName = getLabelCol
    val dataUse = makeDataToUse(dataset, labelName)
    new MulticlassClassificationEvaluator()
      .setLabelCol(labelName)
      .setPredictionCol(getPredictionValueCol)
      .setMetricName(metricName.sparkEntryName)
      .evaluateOrDefault(dataUse, default = default)
  }

}


/**
 * Metrics of MultiClassification Problem
 *
 * @param Precision
 * @param Recall
 * @param F1
 * @param Error
 * @param ThresholdMetrics
 */
case class MultiClassificationMetrics
(
  Precision: Double,
  Recall: Double,
  F1: Double,
  Error: Double,
  ThresholdMetrics: ThresholdMetrics
) extends EvaluationMetrics

/**
 * Threshold-based metrics for multiclass classification
 *
 * Classifications being correct, incorrect, or no classification are defined in terms of the topN and score threshold
 * to be:
 * Correct - score of the true label is in the top N scores AND the score of the true label is >= threshold
 * Incorrect - score of top predicted label >= threshold AND
 * (true label NOT in top N predicted labels OR score of true label < threshold)
 * No prediction - otherwise (score of top predicted label < threshold)
 *
 * @param topNs              list of topN values (used as keys for the count maps)
 * @param thresholds         list of threshold values (correspond to thresholds at the indices
 *                           of the arrays in the count maps)
 * @param correctCounts      map from topN value to an array of counts of correct classifications at each threshold
 * @param incorrectCounts    map from topN value to an array of counts of incorrect classifications at each threshold
 * @param noPredictionCounts map from topN value to an array of counts of no prediction at each threshold
 */
case class ThresholdMetrics
(
  @JsonDeserialize(contentAs = classOf[java.lang.Integer])
  topNs: Seq[Int],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  thresholds: Seq[Double],
  @JsonDeserialize(keyAs = classOf[java.lang.Integer])
  correctCounts: Map[Int, Seq[Long]],
  @JsonDeserialize(keyAs = classOf[java.lang.Integer])
  incorrectCounts: Map[Int, Seq[Long]],
  @JsonDeserialize(keyAs = classOf[java.lang.Integer])
  noPredictionCounts: Map[Int, Seq[Long]]
) extends EvaluationMetrics

