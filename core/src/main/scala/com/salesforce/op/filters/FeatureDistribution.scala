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

package com.salesforce.op.filters

import java.util.Objects

import com.salesforce.op.features.{FeatureDistributionLike, FeatureDistributionType}
import com.salesforce.op.stages.impl.feature.{HashAlgorithm, Inclusion, NumericBucketizer, TextStats}
import com.salesforce.op.utils.json.EnumEntrySerializer
import com.twitter.algebird.Monoid._
import com.twitter.algebird._
import com.twitter.algebird.Operators._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.test.ChiSqTestResult
import org.json4s.jackson.Serialization
import org.json4s.{DefaultFormats, Formats}

import scala.util.Try

/**
 * Class containing summary information for a feature
 *
 * @param name         name of the feature
 * @param key          map key associated with distribution (when the feature is a map)
 * @param count        total count of feature seen
 * @param nulls        number of empties seen in feature
 * @param distribution binned counts of feature values (hashed for strings, evenly spaced bins for numerics)
 * @param summaryInfo  either min and max number of tokens for text data, or splits used for bins for numeric data
 * @param `type`       feature distribution type: training or scoring
 */
case class FeatureDistribution
(
  name: String,
  key: Option[String],
  count: Long,
  nulls: Long,
  distribution: Array[Double],
  summaryInfo: Array[Double],
  moments: Option[Moments] = None,
  cardEstimate: Option[TextStats] = None,
  `type`: FeatureDistributionType = FeatureDistributionType.Training
) extends FeatureDistributionLike {

  /**
   * Get feature key associated to this distribution
   */
  def featureKey: FeatureKey = (name, key)

  /**
   * Check that feature distributions belong to the same feature, key and type.
   *
   * @param fd distribution to compare to
   */
  private def checkMatch(fd: FeatureDistribution): Unit = {
    def check[T](field: String, v1: T, v2: T): Unit = require(v1 == v2,
      s"$field must match to compare or combine feature distributions: $v1 != $v2"
    )
    check("Name", name, fd.name)
    check("Key", key, fd.key)
  }

  /**
   * Get fill rate of feature
   *
   * @return fraction of data that is non empty
   */
  def fillRate(): Double = if (count == 0L) 0.0 else (count - nulls) / count.toDouble

  def topKCardRatio(): Option[Double] = cardEstimate match {
    case Some(x) =>
      val counts = x.valueCounts.values.toList.sortWith(_ > _)
      if (counts.size > 100) {
        Some(counts.take(100).sum / count)
      }
      else {
        Some(counts.sum / count)
      }
   case _ => None
  }

  def cardSize(): Option[Double] = cardEstimate match {
    case Some(x) => Some(x.valueCounts.size)
    case _ => None
  }

  // average number of token per row
  def avgcardCount(): Option[Double] = cardEstimate match {
    case Some(x) => Some(x.valueCounts.values.sum / count)
    case _ => None
  }

  // highest token count ?
  def maxcardCount(): Option[Double] = cardEstimate match {
    case Some(x) => Some(x.valueCounts.values.max)
    case _ => None
  }

  def chiSqUnifTestHash(): ChiSqTestResult = {
    val vectorizedDistr = Vectors.dense(distribution)
    return Statistics.chiSqTest(vectorizedDistr)
  }

  def chiSqUnifTestCard(): (Option[Double], Option[Double]) = cardEstimate match {
    case Some(x) => {
      val vectorizedDistr = Vectors.dense(x.valueCounts.values.toArray.map(_.toDouble))
      val testResult = Statistics.chiSqTest(vectorizedDistr)
      (Some(testResult.statistic), Some(testResult.pValue))
    }
    case _ => (None, None)
  }
  /**
   * Combine feature distributions
   *
   * @param fd other feature distribution (from the same feature)
   * @return summed distribution information
   */
  def reduce(fd: FeatureDistribution): FeatureDistribution = {
    checkMatch(fd)
    // should move this somewhere else
    implicit val testStatsMonoid: Monoid[TextStats] = TextStats.monoid(RawFeatureFilter.MaxCardinality)
    implicit val opMonoid = optionMonoid[TextStats]

    val combinedDist = distribution + fd.distribution
    // summary info can be empty or min max if hist is empty but should otherwise match so take the longest info
    val combinedSummaryInfo = if (summaryInfo.length > fd.summaryInfo.length) summaryInfo else fd.summaryInfo

    val combinedMoments = moments + fd.moments
    val combinedCard = cardEstimate + fd.cardEstimate

    FeatureDistribution(name, key, count + fd.count, nulls + fd.nulls, combinedDist,
      combinedSummaryInfo, combinedMoments, combinedCard, `type`)
  }

  /**
   * Ratio of fill rates between the two distributions symetric with larger value on the top
   *
   * @param fd feature distribution to compare to
   * @return ratio of fill rates
   */
  def relativeFillRatio(fd: FeatureDistribution): Double = {
    checkMatch(fd)
    val (thisFill, thatFill) = (fillRate(), fd.fillRate())
    val (small, large) = if (thisFill < thatFill) (thisFill, thatFill) else (thatFill, thisFill)
    if (small == 0.0) Double.PositiveInfinity else large / small
  }

  /**
   * Absolute difference in empty rates
   *
   * @param fd feature distribution to compare to
   * @return absolute difference of rates
   */
  def relativeFillRate(fd: FeatureDistribution): Double = {
    checkMatch(fd)
    math.abs(fillRate() - fd.fillRate())
  }

  /**
   * Jensen-Shannon divergence from this distribution to the other distribution fed in
   *
   * @param fd other feature distribution
   * @return the KL divergence
   */
  def jsDivergence(fd: FeatureDistribution): Double = {
    checkMatch(fd)
    val combinedCounts = distribution.zip(fd.distribution).filterNot{ case (a, b) => a == 0.0 && b == 0.0 }
    val (thisCount, thatCount) =
      combinedCounts.fold[(Double, Double)]((0.0, 0.0)){ case ((a1, b1), (a2, b2)) => (a1 + a2, b1 + b2) }
    val probs = combinedCounts.map{ case (a, b) => a / thisCount -> b / thatCount }
    val meanProb = probs.map{ case (a, b) => (a + b) / 2}
    def log2(x: Double) = math.log10(x) / math.log10(2.0)
    def klDivergence(a: Double, b: Double) = if (a == 0.0) 0.0 else a * log2(a / b)
    probs.zip(meanProb).map{ case ((a, b), m) => 0.5 * klDivergence(a, m) + 0.5 * klDivergence(b, m) }.sum
  }

  override def toString(): String = {
    val valStr = Seq(
      "type" -> `type`.toString,
      "name" -> name,
      "key" -> key,
      "count" -> count.toString,
      "nulls" -> nulls.toString,
      "distribution" -> distribution.mkString("[", ",", "]"),
      "summaryInfo" -> summaryInfo.mkString("[", ",", "]"),
      "cardinality" -> cardEstimate.map(_.toString).getOrElse(""),
      "moments" -> moments.map(_.toString).getOrElse("")
    ).map { case (n, v) => s"$n = $v" }.mkString(", ")

    s"${getClass.getSimpleName}($valStr)"
  }

  override def equals(that: Any): Boolean = that match {
    case FeatureDistribution(`name`, `key`, `count`, `nulls`, d, s, m, c, `type`) =>
      distribution.deep == d.deep && summaryInfo.deep == s.deep &&
        moments == m && cardEstimate == c
    case _ => false
  }

  override def hashCode(): Int = Objects.hashCode(name, key, count, nulls, distribution,
    summaryInfo, moments, cardEstimate, `type`)
}

object FeatureDistribution {

  val MaxBins = 100000

  implicit val semigroup: Semigroup[FeatureDistribution] = new Semigroup[FeatureDistribution] {
    override def plus(l: FeatureDistribution, r: FeatureDistribution): FeatureDistribution = l.reduce(r)
  }

  implicit val formats: Formats = DefaultFormats +
    EnumEntrySerializer.json4s[FeatureDistributionType](FeatureDistributionType)

  /**
   * Feature distributions to json
   *
   * @param fd feature distributions
   * @return json array
   */
  def toJson(fd: Seq[FeatureDistribution]): String = Serialization.write[Seq[FeatureDistribution]](fd)

  /**
   * Feature distributions from json
   *
   * @param json feature distributions json
   * @return feature distributions array
   */
  def fromJson(json: String): Try[Seq[FeatureDistribution]] = Try {
    Serialization.read[Seq[FeatureDistribution]](json)
  }

  /**
   * Facilitates feature distribution retrieval from computed feature summaries
   *
   * @param featureKey      feature key
   * @param summary         feature summary
   * @param value           optional processed sequence
   * @param bins            number of histogram bins
   * @param textBinsFormula formula to compute the text features bin size.
   *                        Input arguments are [[Summary]] and number of bins to use in computing feature distributions
   *                        (histograms for numerics, hashes for strings). Output is the bins for the text features.
   * @param `type`          feature distribution type: training or scoring
   * @return feature distribution given the provided information
   */
  private[op] def fromSummary(
    featureKey: FeatureKey,
    summary: Summary,
    value: Option[ProcessedSeq],
    bins: Int,
    textBinsFormula: (Summary, Int) => Int,
    `type`: FeatureDistributionType
  ): FeatureDistribution = {
    val (name, key) = featureKey
    val (nullCount, (summaryInfo, distribution)) =
      value.map(seq => 0L -> histValues(seq, summary, bins, textBinsFormula))
        .getOrElse(1L -> (Array(summary.min, summary.max, summary.sum, summary.count) -> new Array[Double](bins)))

    val moments = value.map(momentsValues)
    val cardEstimate = value.map(cardinalityValues)

    FeatureDistribution(
      name = name,
      key = key,
      count = 1L,
      nulls = nullCount,
      summaryInfo = summaryInfo,
      distribution = distribution,
      moments = moments,
      cardEstimate = cardEstimate,
      `type` = `type`
    )
  }

  /**
   * Function to calculate the first five central moments of numeric values, or length of tokens for text features
   *
   * @param values          values to calculate moments
   * @return Moments object containing information about moments
   */
  private def momentsValues(values: ProcessedSeq): Moments = {
    val population = values match {
      case Left(seq) => seq.map(x => x.length.toDouble)
      case Right(seq) => seq
    }
    MomentsGroup.sum(population.map(x => Moments(x)))
  }

  /**
   * Function to track frequency of the first $(MaxCardinality) unique values
   * (number for numeric features, token for text features)
   *
   * @param values          values to track distribution / frequency
   * @return TextStats object containing a Map from a value to its frequency (histogram)
   */
  private def cardinalityValues(values: ProcessedSeq): TextStats = {
    val population = values match {
      case Left(seq) => seq
      case Right(seq) => seq.map(_.toString)
    }
    TextStats(population.groupBy(identity).map{case (key, value) => (key, value.size)})
  }

  /**
   * Function to put data into histogram of counts
   *
   * @param values          values to bin
   * @param summary         summary info for feature (max, min, etc)
   * @param bins            number of bins to produce
   * @param textBinsFormula formula to compute the text features bin size.
   *                        Input arguments are [[Summary]] and number of bins to use in computing feature distributions
   *                        (histograms for numerics, hashes for strings). Output is the bins for the text features.
   * @return a pair consisting of response and predictor feature distributions (in this order)
   * @return the bin information and the binned counts
   */
  private def histValues(
    values: ProcessedSeq,
    summary: Summary,
    bins: Int,
    textBinsFormula: (Summary, Int) => Int
  ): (Array[Double], Array[Double]) = values match {
    case Left(seq) =>
      val numBins = textBinsFormula(summary, bins)
      // TODO: creating too many hasher instances may cause problem, efficiency, garbage collection etc
      val hasher =
        new HashingTF(numFeatures = numBins).setBinary(false)
          .setHashAlgorithm(HashAlgorithm.MurMur3.entryName.toLowerCase)
      Array(summary.min, summary.max, summary.sum, summary.count) -> hasher.transform(seq).toArray

    case Right(seq) => // TODO use kernel fit instead of histogram
      if (summary == Summary.empty) {
        Array(summary.min, summary.max) -> seq.toArray // the seq will always be empty in this case
      } else if (summary.min < summary.max) {
        // total number of bins includes one for edge and one for other
        val step = (summary.max - summary.min) / (bins - 2.0)
        val splits = (0 until bins).map(b => summary.min + step * b).toArray
        val binned = seq.map { v =>
          NumericBucketizer.bucketize(
            splits = splits, trackNulls = false, trackInvalid = true,
            splitInclusion = Inclusion.Left, input = Option(v)
          ).toArray
        }
        val hist = binned.fold(new Array[Double](bins))(_ + _)
        splits -> hist
      } else {
        val same = seq.map(v => if (v == summary.max) 1.0 else 0.0).sum
        val other = seq.map(v => if (v != summary.max) 1.0 else 0.0).sum
        Array(summary.min, summary.max, summary.sum, summary.count) -> Array(same, other)
      }
  }
}
