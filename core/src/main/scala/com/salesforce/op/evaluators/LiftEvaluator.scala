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
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.slf4j.LoggerFactory

/**
 * Evaluator Class to calculate Lift metrics for BinaryClassification problems
 * Intended to build a Lift Plot, or with a threshold, evaluate to a numeric
 * value, liftRatio, to determine model fit. See:
 * https://en.wikipedia.org/wiki/Lift_(data_mining)
 *
 * Algorithm for calculating a chart as seen here:
 * https://www.kdnuggets.com/2016/03/lift-analysis-data-scientist-secret-weapon.html
 *
 * @param threshold decision value to categorize score probabilities into predicted labels
 * @param bandFn    function to convert score distribution into score bands
 * @param uid       UID for evaluator
 */
class LiftEvaluator
(
  threshold: Double = LiftMetrics.defaultThreshold,
  bandFn: RDD[Double] => Seq[(Double, Double, String)] = LiftEvaluator.getDefaultScoreBands,
  override val uid: String = UID[OpBinaryClassificationEvaluator]
) extends OpBinaryClassificationEvaluatorBase[LiftMetrics](uid = uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  override val name: EvalMetric = BinaryClassEvalMetrics.LiftMetrics

  /**
   * Default metrics is liftRatio, which is calculated as:
   * (# yes records with score >= threshold / # total records with score >= threshold)
   * / (# yes records overall / # total records overall)
   *
   * @return double value used as spark eval number
   */
  override def getDefaultMetric: LiftMetrics => Double = _.liftRatio

  /**
   * Evaluates entire dataset, pulling out score and label columns
   * into an RDD and return an an instance of LiftMetrics
   *
   * @param dataset data to evaluate
   * @return metrics
   */
  override def evaluateAll(dataset: Dataset[_]): LiftMetrics = {
    val labelColumnName = getLabelCol
    val dataToUse = makeDataToUse(dataset, labelColumnName)
      .select(col(getProbabilityCol), col(labelColumnName).cast(DoubleType)).rdd
    if (dataToUse.isEmpty()) {
      log.warn("The dataset is empty. Returning empty metrics.")
      LiftMetrics.empty
    } else {
      val scoreAndLabels = dataToUse.map {
        case Row(prob: Vector, label: Double) => (prob(1), label)
        case Row(prob: Double, label: Double) => (prob, label)
      }
      evaluateScoreAndLabels(scoreAndLabels)
    }
  }

  /**
   * Calculates a Seq of lift metrics per score band and
   * an overall lift ratio value, returned as LiftMetrics,
   * from an RDD of scores and labels
   *
   * @param scoreAndLabels RDD of score and label doubles
   * @return an instance of LiftMetrics
   */
  def evaluateScoreAndLabels(scoreAndLabels: RDD[(Double, Double)]): LiftMetrics = {
    val liftMetricBands = LiftEvaluator.liftMetricBands(scoreAndLabels, bandFn)
    val overallRate = LiftEvaluator.thresholdLiftRate(scoreAndLabels, 0.0)
    val liftRatio = LiftEvaluator.liftRatio(overallRate, scoreAndLabels, threshold)
    LiftMetrics(
      liftMetricBands = liftMetricBands,
      threshold = threshold,
      liftRatio = liftRatio,
      overallRate = overallRate)
  }
}

/**
 * Object to calculate Lift metrics for BinaryClassification problems
 * Intended to build a Lift Plot.
 *
 * Algorithm for calculating a chart as seen here:
 * https://www.kdnuggets.com/2016/03/lift-analysis-data-scientist-secret-weapon.html
 */
object LiftEvaluator {

  /**
   * Builds Seq of LiftMetricBand using RDD api
   *
   * @param scoreAndLabels RDD[(Double, Double)] of BinaryClassification (score, label) tuples
   * @param getScoreBands  function to calculate score bands, potentially using score distribution
   * @return Seq of LiftMetricBand containers of Lift calculations
   */
  private[op] def liftMetricBands
  (
    scoreAndLabels: RDD[(Double, Double)],
    bandFn: RDD[Double] => Seq[(Double, Double, String)]
  ): Seq[LiftMetricBand] = {
    val scores = scoreAndLabels.map { case (score, _) => score }
    val bands = bandFn(scores)
    val bandedLabels = scoreAndLabels.map { case (score, label) =>
      (categorizeScoreIntoBand((score, bands)), label)
    }.collect { case (Some(band), label) => (band, label) }
    val perBandCounts = aggregateBandedLabels(bandedLabels)
    bands.map { case (lower, upper, band) =>
      formatLiftMetricBand(lower, upper, band, perBandCounts)
    }.sortBy(band => band.lowerBound)
  }

  /**
   * function to return score bands for calculating lift
   * Default: 10 equidistant bands for all 0.1 increments
   * from 0.0 to 1.0
   *
   * @param scores RDD of scores. unused in this function
   * @return sequence of (lowerBound, upperBound, bandString) tuples
   */
  private[op] def getDefaultScoreBands(scores: RDD[Double]): Seq[(Double, Double, String)] =
    Seq(
      (0.0, 0.1, "0-10"),
      (0.1, 0.2, "10-20"),
      (0.2, 0.3, "20-30"),
      (0.3, 0.4, "30-40"),
      (0.4, 0.5, "40-50"),
      (0.5, 0.6, "50-60"),
      (0.6, 0.7, "60-70"),
      (0.7, 0.8, "70-80"),
      (0.8, 0.9, "80-90"),
      (0.9, 1.0, "90-100")
    )

  /**
   * PartialFunction. Defined when scores are [0.0, 1.0]
   * Places a score Double into a score band based on
   * lower and upper bounds
   *
   * @param score BinaryClassification score Double, [0.0, 1.0]
   * @param bands sequence of upper/lower score bands
   * @return optional key to describe categorized band, if found
   */
  private[op] def categorizeScoreIntoBand:
  PartialFunction[(Double, Seq[(Double, Double, String)]), Option[String]] = {
    case (score: Double, bands: Seq[(Double, Double, String)])
      if (score >= 0.0) & (score <= 1.0) =>
      bands.find { case (l, u, _) =>
        (score >= l) & (score <= u)
      } match {
        case Some((_, _, bandString)) => Some(bandString)
        case None => None
      }
  }

  /**
   * aggregates labels into counts by lift band
   *
   * @param bandedLabels PairRDD of (bandString, label)
   * @return Map of bandString -> (total count, count of positive labels)
   */
  private[op] def aggregateBandedLabels
  (
    bandedLabels: RDD[(String, Double)]
  ): Map[String, (Long, Long)] = {
    val countsPerBand = bandedLabels.countByKey()
    val truesPerBand = bandedLabels.aggregateByKey(zeroValue = 0.0)(
      { case (sum, label) => sum + label },
      { case (sumX, sumY) => sumX + sumY })
      .collectAsMap()
    countsPerBand.map { case (band, count) =>
      band -> (count, truesPerBand.getOrElse(band, 0.0).toLong)
    }.toMap
  }

  /**
   * Subsets scoresAndLabels to only records with scores greater than
   * a threshold, categorizing them as predicted "yes" labels, then
   * returns (# of true "yes" labels / # predicted yes)
   *
   * @param scoreAndLabels RDD of labels and scores
   * @param threshold      decision value, where scores >= thres are predicted "yes"
   * @return Optional lift rate, None if denominator is 0
   */
  private[op] def thresholdLiftRate
  (
    scoreAndLabels: RDD[(Double, Double)],
    threshold: Double
  ): Option[Double] = {
    val (yesCount, totalCount) = scoreAndLabels.aggregate(zeroValue = (0L, 0L))({
      case ((yesCount, totalCount), (score, label)) => {
        if (score < threshold) (yesCount, totalCount)
        else (yesCount + label.toLong, totalCount + 1L)
      }
    }, { case ((yesCountX, totalCountX), (yesCountY, totalCountY)) =>
      (yesCountX + yesCountY, totalCountX + totalCountY)
    })
    totalCount match {
      case 0L => None
      case _ => Some(yesCount.toDouble / totalCount.toDouble)
    }
  }

  /**
   * Given a threshold decision value, calculates the lift
   * in label prediction accuracy over random, as described here:
   * https://en.wikipedia.org/wiki/Lift_(data_mining)
   *
   * @param overallRate    # yes / total count across all data
   * @param scoreAndLabels RDD of scores and labels
   * @param threshold      decision boundary for categorizing scores
   * @return lift ratio, given threshold
   */
  private[op] def liftRatio
  (
    overallRate: Option[Double],
    scoreAndLabels: RDD[(Double, Double)],
    threshold: Double
  ): Double = overallRate match {
    case None => LiftMetrics.defaultLiftRatio
    case Some(0.0) => LiftMetrics.defaultLiftRatio
    case Some(rate) => {
      val thresholdLift = thresholdLiftRate(scoreAndLabels, threshold)
      thresholdLift match {
        case None => LiftMetrics.defaultLiftRatio
        case Some(thresholdRate) => thresholdRate / rate
      }
    }
  }

  /**
   * Formats lift data in one band into LiftMetricBand data,
   * including lower bound of score band, upper bound, total record
   * count per band, and lift (# trues / total)
   *
   * @param lower         lower bound of band
   * @param upper         upper bound of band
   * @param bandString    String key of band e.g. "10-20"
   * @param perBandCounts calculated total counts and counts of true labels
   * @return LiftMetricBand container of metrics
   */
  private[op] def formatLiftMetricBand
  (
    lower: Double,
    upper: Double,
    bandString: String,
    perBandCounts: Map[String, (Long, Long)]
  ): LiftMetricBand = {
    perBandCounts.get(bandString) match {
      case Some((numTotal, numYes)) => {
        val lift = numTotal match {
          case 0L => None
          case _ => Some(numYes.toDouble / numTotal)
        }
        LiftMetricBand(
          group = bandString,
          lowerBound = lower,
          upperBound = upper,
          rate = lift,
          totalCount = numTotal,
          yesCount = numYes,
          noCount = numTotal - numYes
        )
      }
      case None => LiftMetricBand(
        group = bandString,
        lowerBound = lower,
        upperBound = upper,
        rate = None,
        totalCount = 0L,
        yesCount = 0L,
        noCount = 0L
      )
    }
  }

}

/**
 * Stores basic lift values for a specific band of scores
 *
 * @param group      name / key for score band
 * @param lowerBound minimum score represented in lift
 * @param upperBound maximum score represented in lift
 * @param rate       optional calculated lift value, i.e. # yes / total count
 * @param totalCount total number of records in score band
 * @param yesCount   number of yes records in score band
 * @param noCount    number of no records in score band
 */
case class LiftMetricBand
(
  group: String,
  lowerBound: Double,
  upperBound: Double,
  rate: Option[Double],
  totalCount: Long,
  yesCount: Long,
  noCount: Long
) extends EvaluationMetrics

/**
 * Stores sequence of lift score band metrics
 * as well as overall lift values
 *
 * @param liftMetricBands Seq of LiftMetricBand, calculated by LiftEvaluator
 * @param threshold       threshold used to categorize scores
 * @param liftRatio       overall lift ratio, given a specified threshold
 * @param overallRate     # yes records / # total records
 */
case class LiftMetrics
(
  @JsonDeserialize(contentAs = classOf[LiftMetricBand])
  liftMetricBands: Seq[LiftMetricBand],
  threshold: Double,
  liftRatio: Double,
  overallRate: Option[Double]
) extends EvaluationMetrics

/**
 * Companion object to LiftMetrics case class
 * for storing default values
 */
object LiftMetrics {
  val defaultThreshold = 0.5
  val defaultLiftRatio = 1.0

  def empty: LiftMetrics = LiftMetrics(Seq(), defaultThreshold, defaultLiftRatio, None)
}
