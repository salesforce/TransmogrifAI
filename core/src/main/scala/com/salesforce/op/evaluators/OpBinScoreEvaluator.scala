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
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row}
import org.slf4j.LoggerFactory
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import com.twitter.algebird.Operators._
import com.twitter.algebird.Monoid._
import org.apache.spark.rdd.RDD

/**
 *
 * Evaluator for Binary Classification which provides statistics about the predicted scores.
 * This evaluator creates the specified number of bins and computes the statistics for each bin
 * and returns BinaryClassificationBinMetrics, which contains
 *
 * Total number of data points per bin
 * Average Score per bin
 * Average Conversion rate per bin
 * Bin Centers for each bin
 * BrierScore for the overall dataset is also computed, which is a default metric as well.
 *
 * @param name            name of default metric
 * @param isLargerBetter  is metric better if larger
 * @param uid             uid for instance
 */
private[op] class OpBinScoreEvaluator
(
  override val name: EvalMetric = OpEvaluatorNames.BinScore,
  override val isLargerBetter: Boolean = true,
  override val uid: String = UID[OpBinScoreEvaluator],
  val numBins: Int = 100
) extends OpBinaryClassificationEvaluatorBase[BinaryClassificationBinMetrics](uid = uid) {

  require(numBins > 0, "numBins must be positive")
  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: BinaryClassificationBinMetrics => Double = _.brierScore

  override def evaluateAll(data: Dataset[_]): BinaryClassificationBinMetrics = {
    val labelColumnName = getLabelCol
    val dataProcessed = makeDataToUse(data, labelColumnName)

    val rdd = dataProcessed.select(col(getProbabilityCol), col(labelColumnName).cast(DoubleType)).rdd
    if (rdd.isEmpty()) {
      log.error("The dataset is empty. Returning empty metrics")
      BinaryClassificationBinMetrics(0.0, Seq(), Seq(), Seq(), Seq())
    } else {
      val scoreAndLabels = rdd.map {
        case Row(prob: Vector, label: Double) => (prob(1), label)
        case Row(prob: Double, label: Double) => (prob, label)
      }

      // Finding stats per bin -> avg score, avg conv rate,
      // total num of data points and overall brier score.
      val stats = scoreAndLabels.map {
        case (score, label) => (getBinIndex(score), (score, label, 1L, math.pow((score - label), 2)))
      }.reduceByKey(_ + _).map {
        case (bin, (scoreSum, labelSum, count, squaredError)) =>
          (bin, scoreSum / count, labelSum / count, count, squaredError)
      }.collect()

      val (averageScore, averageConversionRate, numberOfDataPoints, brierScoreSum, numberOfPoints) =
        stats.foldLeft((new Array[Double](numBins), new Array[Double](numBins), new Array[Long](numBins), 0.0, 0L)) {
          case ((score, convRate, dataPoints, brierScoreSum, totalPoints),
          (binIndex, avgScore, avgConvRate, counts, squaredError)) => {

            score(binIndex) = avgScore
            convRate(binIndex) = avgConvRate
            dataPoints(binIndex) = counts

            (score, convRate, dataPoints, brierScoreSum + squaredError, totalPoints + counts)
          }
        }

      // binCenters is the center point in each bin.
      // e.g., for bins [(0.0 - 0.5), (0.5 - 1.0)], bin centers are [0.25, 0.75].
      val binCenters = (for {i <- 0 to numBins} yield ((i + 0.5) / numBins)).dropRight(1)

      val metrics = BinaryClassificationBinMetrics(
        brierScore = brierScoreSum / numberOfPoints,
        binCenters = binCenters,
        numberOfDataPoints = numberOfDataPoints,
        averageScore = averageScore,
        averageConversionRate = averageConversionRate
      )

      log.info("Evaluated metrics: {}", metrics.toString)
      metrics
    }
  }

  // getBinIndex finds which bin the score associates with.
  private def getBinIndex(score: Double): Int = {
    val binIndex = math.min(numBins - 1, (score * numBins).toInt)
    math.max(binIndex, 0) // if score is negative, assign it to bin 0.
  }
}

/**
 * Metrics of BinaryClassificationBinMetrics
 *
 * @param binCenters            center of each bin
 * @param numberOfDataPoints    total number of data points in each bin
 * @param averageScore          average score in each bin
 * @param averageConversionRate average conversion rate in each bin
 * @param brierScore            brier score for overall dataset
 */
case class BinaryClassificationBinMetrics
(
  brierScore: Double,
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  binCenters: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Long])
  numberOfDataPoints: Seq[Long],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  averageScore: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  averageConversionRate: Seq[Double]
) extends EvaluationMetrics
