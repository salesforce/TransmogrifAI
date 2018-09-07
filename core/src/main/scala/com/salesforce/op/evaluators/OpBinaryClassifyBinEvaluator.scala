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

import com.salesforce.op.UID
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.DoubleType
import org.slf4j.LoggerFactory
import org.apache.spark.Partitioner

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
private[op] class OpBinaryClassifyBinEvaluator
(
  override val name: EvalMetric = OpEvaluatorNames.Binary,
  override val isLargerBetter: Boolean = true,
  override val uid: String = UID[BinaryClassificationBinMetrics],
  val numBins: Int = 100
) extends OpBinaryClassificationEvaluatorBase[BinaryClassificationBinMetrics](uid = uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: BinaryClassificationBinMetrics => Double = _.BrierScore

  override def evaluateAll(data: Dataset[_]): BinaryClassificationBinMetrics = {
    val labelColName = getLabelCol
    val dataUse = makeDataToUse(data, labelColName)

    val (rawPredictionColName, predictionColName, probabilityColName) =
      (getRawPredictionCol, getPredictionValueCol, getProbabilityCol)
    log.debug(
      "Evaluating metrics on columns :\n label : {}\n rawPrediction : {}\n prediction : {}\n probability : {}\n",
      labelColName, rawPredictionColName, predictionColName, probabilityColName
    )

    import dataUse.sparkSession.implicits._
    val rdd = dataUse.select(predictionColName, labelColName).as[(Double, Double)].rdd

    if (rdd.isEmpty()) {
      log.error("The dataset is empty")
      BinaryClassificationBinMetrics(0.0, Seq.empty[Double], Seq.empty[Long], Seq.empty[Double], Seq.empty[Double])
    } else {
      val scoreAndLabels =
        dataUse.select(col(probabilityColName), col(labelColName).cast(DoubleType)).rdd.map {
          case Row(prob: Vector, label: Double) => (prob(1), label)
          case Row(prob: Double, label: Double) => (prob, label)
        }

      if (numBins == 0) {
        log.error("numBins is set to 0. Returning empty metrics")
        BinaryClassificationBinMetrics(0.0, Seq.empty[Double], Seq.empty[Long], Seq.empty[Double], Seq.empty[Double])
      } else {
        // Find the significant digit to which the scores needs to be rounded, based of numBins.
        val significantDigitToRoundOff = math.log10(numBins).toInt + 1
        val scoreAndLabelsRounded = for {i <- scoreAndLabels}
          yield (BigDecimal(i._1).setScale(significantDigitToRoundOff,
            BigDecimal.RoundingMode.HALF_UP).toDouble, (i._1, i._2))

        // Create `numBins` bins and place each score in its corresponding bin.
        val binnedValues = scoreAndLabelsRounded.partitionBy(new OpBinPartitioner(numBins)).values

        // compute the average score per bin
        val averageScore = binnedValues.mapPartitions(scores => {
          val (totalScore, count) = scores.foldLeft(0.0, 0)(
            (r: (Double, Int), s: (Double, Double)) => (r._1 + s._1, r._2 + 1))
          Iterator(if (count == 0) 0.0 else totalScore / count)
        }).collect().toSeq

        // compute the average conversion rate per bin. Convertion rate is the number of 1's in labels.
        val averageConvertionRate = binnedValues.mapPartitions(scores => {
          val (totalConversion, count) = scores.foldLeft(0.0, 0)(
            (r: (Double, Int), s: (Double, Double)) => (r._1 + s._2, r._2 + 1))
          Iterator(if (count == 0) 0.0 else totalConversion / count)
        }).collect().toSeq

        // compute total number of data points in each bin.
        val numberOfDataPoints = binnedValues.mapPartitions(scores => Iterator(scores.length.toLong)).collect().toSeq

        // binCenters is the center point in each bin.
        // e.g., for bins [(0.0 - 0.5), (0.5 - 1.0)], bin centers are [0.25, 0.75].
        val binCenters = (for {i <- 0 to numBins} yield ((i + 0.5) / numBins)).dropRight(1)

        // brier score of entire dataset.
        val brierScore = scoreAndLabels.map { case (score, label) => math.pow((score - label), 2) }.mean()

        val metrics = BinaryClassificationBinMetrics(
          BrierScore = brierScore,
          BinCenters = binCenters,
          NumberOfDataPoints = numberOfDataPoints,
          AverageScore = averageScore,
          AverageConversionRate = averageConvertionRate
        )

        log.info("Evaluated metrics: {}", metrics.toString)
        metrics
      }
    }
  }
}

// BinPartitioner which partition the bins.
class OpBinPartitioner(override val numPartitions: Int) extends Partitioner {

  // computes the bin number(0-indexed) to which the datapoint is assigned.
  // For Score 1.0, overflow happens. So, use math.min(last_bin, bin_index__computed).
  def getPartition(key: Any): Int = key match {
    case score: Double => math.min(numPartitions - 1, (score * numPartitions).toInt)
  }
}

/**
 * Metrics of BinaryClassificationBinMetrics
 *
 * @param BinCenters            center of each bin
 * @param NumberOfDataPoints    total number of data points in each bin
 * @param AverageScore          average score in each bin
 * @param AverageConversionRate average conversion rate in each bin
 * @param BrierScore            brier score for overall dataset
 */
case class BinaryClassificationBinMetrics
(
  BrierScore: Double,
  BinCenters: Seq[Double],
  NumberOfDataPoints: Seq[Long],
  AverageScore: Seq[Double],
  AverageConversionRate: Seq[Double]
) extends EvaluationMetrics
