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

import org.apache.spark.rdd.RDD

/**
 * Object to calculate Lift metrics for BinaryClassification problems
 * Intended to build a Lift Plot.
 *
 * Algorithm for calculating a chart as seen here:
 * https://www.kdnuggets.com/2016/03/lift-analysis-data-scientist-secret-weapon.html
 */
object LiftEvaluator {

  /**
   * Stores basic lift values for a specific band of scores
   *
   * @param group      name / key for score band
   * @param lowerBound minimum score represented in lift
   * @param upperBound maximum score represented in lift
   * @param rate       optional calculated lift value, i.e. # yes / total count
   * @param average    optional lift rate across all score bands
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
    average: Option[Double],
    totalCount: Long,
    yesCount: Long,
    noCount: Long
  ) extends EvaluationMetrics

  /**
   * Builds Seq[LiftMetricBand] for BinaryClassificationMetrics, calls liftMetricBands function
   * with default score bands function
   *
   * @param scoreAndLabels RDD[(Double, Double)] of BinaryClassification (score, label) tuples
   * @return Seq of LiftMetricBand containers of Lift calculations
   */
  def apply
  (
    scoreAndLabels: RDD[(Double, Double)]
  ): Seq[LiftMetricBand] = {
    liftMetricBands(
      scoreAndLabels,
      getDefaultScoreBands
    )
  }

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
    getScoreBands: RDD[Double] => Seq[(Double, Double, String)]
  ): Seq[LiftMetricBand] = {
    val bands = getScoreBands(scoreAndLabels.map { case (score, _) => score })
    val bandedLabels = scoreAndLabels.map { case (score, label) =>
      (categorizeScoreIntoBand((score, bands)), label)
    }.collect { case (Some(band), label) => (band, label) }
    val perBandCounts = aggregateBandedLabels(bandedLabels)
    val overallRate = overallLiftRate(perBandCounts)
    bands.map({ case (lower, upper, band) =>
      formatLiftMetricBand(lower, upper, band, perBandCounts, overallRate)
    }).sortBy(band => band.lowerBound)
  }

  /**
   * function to return score bands for calculating lift
   * Default: 10 equidistant bands for all 0.1 increments
   * from 0.0 to 1.0
   *
   * @param scores RDD of scores. unused in this function
   * @return sequence of (lowerBound, upperBound, bandString) tuples
   */
  private[op] def getDefaultScoreBands(scores: RDD[Double]):
  Seq[(Double, Double, String)] =
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
   * calculates a baseline "yes" rate across score bands
   *
   * @param perBandCounts
   * @return overall # yes / total records across all bands
   */
  private[op] def overallLiftRate(perBandCounts: Map[String, (Long, Long)]): Option[Double] = {
    val overallTotalCount = perBandCounts.values.map({ case (totalCount, _) => totalCount }).sum
    val overallYesCount = perBandCounts.values.map({ case (_, yesCount) => yesCount }).sum
    overallTotalCount match {
      case 0L => None
      case _ => Some(overallYesCount.toDouble / overallTotalCount)
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
   * @param overallRate   optional overall Lift rate across all bands
   * @return LiftMetricBand container of metrics
   */
  private[op] def formatLiftMetricBand
  (
    lower: Double,
    upper: Double,
    bandString: String,
    perBandCounts: Map[String, (Long, Long)],
    overallRate: Option[Double]
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
          average = overallRate,
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
        average = overallRate,
        totalCount = 0L,
        yesCount = 0L,
        noCount = 0L
      )
    }
  }

}
