package com.salesforce.op.evaluators

import org.apache.spark.rdd.RDD

/**
  * Object to calculate Lift metrics for BinaryClassification problems
  * Intended for write-back to core for Scorecard or to Looker
  *
  * Algorithm for calculating a chart as seen here:
  * https://www.kdnuggets.com/2016/03/lift-analysis-data-scientist-secret-weapon.html
  */
object LiftEvaluator {

  /**
    * Scoreband name that represents the lift values across
    * all scores, 0 to 100
    */
  private[op] val overallScoreband = "overall"

  /**
    * Stores basic lift values for a specific band of scores
    *
    * @param group   name / key for score band
    * @param lowerBound minimum score represented in lift
    * @param upperBound maximum score represented in lift
    * @param rate calculated lift value, i.e. # yes / total count
    * @param average lift rate across all score bands
    * @param totalCount  total number of records in score band
    * @param yesCount number of yes records in score band
    * @param noCount number of no records in score band
    */
  case class LiftMetricBand
  (
    group: String,
    lowerBound: Double,
    upperBound: Double,
    rate: Double,
    average: Double,
    totalCount: Long,
    yesCount: Long,
    noCount: Long
  )

  /**
    * Builds Lift Map for serialization, wrapper for liftMap function
    * for the DataFrame api
    *
    * @param holdoutDF DataFrame of scored hold-out data
    * @param labelCol  column name for labels
    * @param scoreCol  column name for scores
    * @return AutoMLMetrics: Metrics object for serialization
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
    * Builds Lift Map for serialization using RDD api
    *
    * @param labelsAndScores RDD[(Double, Double)] of BinaryClassification (label, score) tuples
    * @return Seq of LiftMetricBand containers of Lift calculations
    */
  private[op] def liftMetricBands
  (
    scoreAndLabels: RDD[(Double, Double)],
    getScoreBands: RDD[Double] => Seq[(Double, Double, String)]
  ): Seq[LiftMetricBand] = {
    val bands = getScoreBands(scoreAndLabels.map{case (score, _) => score})
    val bandedLabels = scoreAndLabels.map { case (score, label) =>
      (categorizeScoreIntoBand((score, bands)), label)
    }.collect { case (Some(band), label) => (band, label) }
    val perBandCounts = aggregateBandedLabels(bandedLabels)
    val overallRate = overallLiftRate(perBandCounts)
    bands.map({ case (lower, upper, band) =>
      formatLiftMetricBand(lower, upper, band, perBandCounts, overallRate)
    })
  }

  /**
    * function to return score bands for calculating lift
    * Default: 10 equidistant bands for all 0.1 increments
    * from 0.0 to 1.0
    *
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
    *
    *
    * @param perBandCounts
    * @return
    */
  private[op] def overallLiftRate(perBandCounts: Map[String, (Long, Long)]): Double = {
    val overallTotalCount = perBandCounts.values.map({case (totalCount, _) => totalCount}).sum
    val overallYesCount = perBandCounts.values.map({case (_, yesCount) => yesCount}).sum
    overallTotalCount match {
      case 0L => Double.NaN
      case _ => overallYesCount.toDouble / overallTotalCount
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
    perBandCounts: Map[String, (Long, Long)],
    overallRate: Double
  ): LiftMetricBand = {
    perBandCounts.get(bandString) match {
      case Some((numTotal, numYes)) => {
        val lift = numTotal match {
          case 0.0 => Double.NaN
          case _ => numYes.toDouble / numTotal
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
        rate = Double.NaN,
        average = overallRate,
        totalCount = 0L,
        yesCount = 0L,
        noCount = 0L
      )
    }
  }


}
