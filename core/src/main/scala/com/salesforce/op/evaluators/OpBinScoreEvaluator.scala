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
import com.twitter.algebird.Operators._
import com.twitter.algebird.Tuple4Semigroup
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}
import org.slf4j.LoggerFactory

/**
 *
 * Evaluator for Binary Classification which provides statistics about the predicted scores.
 * This evaluator creates the specified number of bins and computes the statistics for each bin
 * and returns [[BinaryClassificationBinMetrics]].
 *
 * @param numOfBins       number of bins to produce
 * @param isLargerBetter  false, i.e. larger BrierScore values are not better
 * @param uid             uid for instance
 */
private[op] class OpBinScoreEvaluator
(
  val numOfBins: Int = 10,
  override val isLargerBetter: Boolean = false,
  uid: String = UID[OpBinScoreEvaluator]
) extends OpBinaryClassificationEvaluatorBase[BinaryClassificationBinMetrics](uid = uid) {

  override val name: EvalMetric = OpEvaluatorNames.BinScore

  require(numOfBins > 0, "numOfBins must be positive")
  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: BinaryClassificationBinMetrics => Double = _.BrierScore

  def evaluateAll(data: Dataset[_]): BinaryClassificationBinMetrics = {
    val labelColumnName = getLabelCol
    val dataToUse = makeDataToUse(data, labelColumnName)
      .select(col(getProbabilityCol), col(labelColumnName).cast(DoubleType)).rdd
    val scoreAndLabels = dataToUse.map {
      case Row(prob: Vector, label: Double) => (prob(1), label)
      case Row(prob: Double, label: Double) => (prob, label)
    }
    evaluateScoreAndLabels(scoreAndLabels)
  }

  def evaluateScoreAndLabels(scoreAndLabels: RDD[(Double, Double)]): BinaryClassificationBinMetrics = {

    val (maxScore, minScore) = scoreAndLabels.map {
      case (score, _) => (score, score)
    }.fold(1.0, 0.0) {
      case ((maxVal, minVal), (scoreMax, scoreMin)) =>
        (math.max(maxVal, scoreMax), math.min(minVal, scoreMin))
    }

    // Finding stats per bin -> avg score, avg conv rate,
    // total num of data points and overall brier score.
    implicit val sg = new Tuple4Semigroup[Double, Double, Long, Double]()
    val stats = scoreAndLabels.map {
      case (score, label) =>
        (getBinIndex(score, minScore, maxScore), (score, label, 1L, math.pow(score - label, 2)))
    }.reduceByKey(_ + _).map {
      case (bin, (scoreSum, labelSum, count, squaredError)) =>
        (bin, scoreSum, labelSum, count, squaredError)
    }.collect()

    stats.toList match {
      case Nil => BinaryClassificationBinMetrics.empty
      case _ => {
        val zero = (new Array[Double](numOfBins), new Array[Double](numOfBins),
          new Array[Long](numOfBins), new Array[Double](numOfBins), 0.0, 0L)
        val (averageScore, averageConversionRate, numberOfDataPoints, sumOfLabels, brierScoreSum, numberOfPoints) =
          stats.foldLeft(zero) {
            case ((score, convRate, dataPoints, labelSums, brierScoreSum, totalPoints),
            (binIndex, scoreSum, labelSum, counts, squaredError)) =>
              score(binIndex) = scoreSum / counts
              convRate(binIndex) = labelSum / counts
              dataPoints(binIndex) = counts
              labelSums(binIndex) = labelSum
              (score, convRate, dataPoints, labelSums, brierScoreSum + squaredError, totalPoints + counts)
          }

        // binCenters is the center point in each bin.
        // e.g., for bins [(0.0 - 0.5), (0.5 - 1.0)], bin centers are [0.25, 0.75].
        val diff = maxScore - minScore
        val binCenters = for {i <- 0 until numOfBins} yield minScore + ((diff * i) / numOfBins) + (diff / (2 * numOfBins))

        val metrics = BinaryClassificationBinMetrics(
          BrierScore = brierScoreSum / numberOfPoints,
          binSize = diff / numOfBins,
          binCenters = binCenters,
          numberOfDataPoints = numberOfDataPoints,
          sumOfLabels = sumOfLabels,
          averageScore = averageScore,
          averageConversionRate = averageConversionRate
        )

        log.info("Evaluated metrics: {}", metrics.toString)
        metrics
      }
    }
  }

  // getBinIndex finds which bin the score associates with.
  private def getBinIndex(score: Double, minScore: Double, maxScore: Double): Int = {
    val binIndex = numOfBins * (score - minScore) / (maxScore - minScore)
    math.min(numOfBins - 1, binIndex.toInt)
  }
}

/**
 * Metrics of BinaryClassificationBinMetrics
 *
 * @param BrierScore            brier score for overall dataset
 * @param binSize               size of each bin
 * @param binCenters            center of each bin
 * @param numberOfDataPoints    total number of data points in each bin
 * @param
 * @param averageScore          average score in each bin
 * @param averageConversionRate average conversion rate in each bin
 */
case class BinaryClassificationBinMetrics
(
  BrierScore: Double,
  binSize: Double,
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  binCenters: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Long])
  numberOfDataPoints: Seq[Long],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  sumOfLabels: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  averageScore: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  averageConversionRate: Seq[Double]
) extends EvaluationMetrics

object BinaryClassificationBinMetrics {
  def empty: BinaryClassificationBinMetrics = BinaryClassificationBinMetrics(0.0, 0.0, Seq(), Seq(), Seq(), Seq(), Seq())
}
