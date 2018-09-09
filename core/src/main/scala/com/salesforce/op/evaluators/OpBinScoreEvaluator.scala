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
import org.apache.spark.Partitioner
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import com.twitter.algebird.Operators._

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

    import dataProcessed.sparkSession.implicits._

    if (dataProcessed.select(getPredictionValueCol, labelColumnName).as[(Double, Double)].rdd.isEmpty()) {
      log.error("The dataset is empty. Returning empty metrics")
      BinaryClassificationBinMetrics(0.0, Seq(), Seq(), Seq(), Seq())
    } else {
      val scoreAndLabels = dataProcessed.select(col(getProbabilityCol), col(labelColumnName).cast(DoubleType)).rdd.map {
        case Row(prob: Vector, label: Double) => (prob(1), label)
        case Row(prob: Double, label: Double) => (prob, label)
      }

      val stats = scoreAndLabels.map {
        case (score, label) => (getBinIndex(score), (score, label, 1L))
      }.reduceByKeyLocally(_ + _).map {
        case (bin, (scoreSum, labelSum, count)) => (bin, (scoreSum / count, labelSum / count, count))
      }

      // For the bins, which don't have any scores, fill 0's.
      val statsForBins = {
        for {i <- 0 to numBins - 1}  yield stats.getOrElse(i, (0.0, 0.0, 0L))
      }

      val averageScore = statsForBins.map(_._1)
      val averageConversionRate = statsForBins.map(_._2)
      val numberOfDataPoints = statsForBins.map(_._3)

      // binCenters is the center point in each bin.
      // e.g., for bins [(0.0 - 0.5), (0.5 - 1.0)], bin centers are [0.25, 0.75].
      val binCenters = (for {i <- 0 to numBins} yield ((i + 0.5) / numBins)).dropRight(1)

      // brier score of entire dataset.
      val brierScore = scoreAndLabels.map { case (score, label) => math.pow((score - label), 2) }.mean()

      val metrics = BinaryClassificationBinMetrics(
        brierScore = brierScore,
        binCenters = binCenters,
        numberOfDataPoints = numberOfDataPoints,
        averageScore = averageScore,
        averageConversionRate = averageConversionRate
      )

      log.info("Evaluated metrics: {}", metrics.toString)
      metrics
    }
  }

   private def getBinIndex(score: Double): Int = {
    math.min(numBins - 1, (score * numBins).toInt)
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
