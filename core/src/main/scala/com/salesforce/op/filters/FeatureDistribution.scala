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

import com.salesforce.op.stages.impl.feature.{HashAlgorithm, Inclusion, NumericBucketizer}
import com.salesforce.op.features.FeatureDistributionLike
import com.salesforce.op.stages.impl.feature.{Inclusion, NumericBucketizer}
import com.twitter.algebird.Semigroup
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import org.apache.spark.mllib.feature.HashingTF

/**
 * Class containing summary information for a feature
 *
 * @param name name of the feature
 * @param key map key associated with distribution (when the feature is a map)
 * @param count total count of feature seen
 * @param nulls number of empties seen in feature
 * @param distribution binned counts of feature values (hashed for strings, evently spaced bins for numerics)
 * @param summaryInfo either min and max number of tokens for text data,
 *                    or number of splits used for bins for numeric data
 */
case class FeatureDistribution
(
  name: String,
  key: Option[String],
  count: Long,
  nulls: Long,
  distribution: Array[Double],
  summaryInfo: Array[Double]
) extends FeatureDistributionLike {

  /**
   * Get feature key associated to this distribution
   */
  def featureKey: FeatureKey = (name, key)

  /**
   * Check that feature distributions belong to the same feature and key.
   *
   * @param fd distribution to compare to
   */
  def checkMatch(fd: FeatureDistribution): Unit =
    assert(name == fd.name && key == fd.key, "Name and key must match to compare or combine FeatureDistribution")

  /**
   * Get fill rate of feature
   *
   * @return fraction of data that is non empty
   */
  def fillRate(): Double = if (count == 0L) 0.0 else (count - nulls) / count.toDouble

  /**
   * Combine feature distributions
   *
   * @param fd other feature distribution (from the same feature)
   * @return summed distribution information
   */
  def reduce(fd: FeatureDistribution): FeatureDistribution = {
    checkMatch(fd)
    val combinedDist = distribution + fd.distribution
    // summary info can be empty or min max if hist is empty but should otherwise match so take the longest info
    val combinedSummary = if (summaryInfo.length > fd.summaryInfo.length) summaryInfo else fd.summaryInfo
    FeatureDistribution(name, key, count + fd.count, nulls + fd.nulls, combinedDist, combinedSummary)
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
    val (thisCount, thatCount) = combinedCounts
      .fold[(Double, Double)]( (0, 0)){ case ((a1, b1), (a2, b2)) => (a1 + a2, b1 + b2) }
    val probs = combinedCounts.map{ case (a, b) => a / thisCount -> b / thatCount }
    val meanProb = probs.map{ case (a, b) => (a + b) / 2}
    def log2(x: Double) = math.log10(x) / math.log10(2.0)
    def klDivergence(a: Double, b: Double) = if (a == 0.0) 0.0 else a * log2(a / b)
    probs.zip(meanProb).map{ case ((a, b), m) => 0.5 * klDivergence(a, m) + 0.5 * klDivergence(b, m) }.sum
  }

  override def toString(): String = {
    s"Name=$name, Key=$key, Count=$count, Nulls=$nulls, Histogram=${distribution.toList}, BinInfo=${summaryInfo.toList}"
  }
}

private[op] object FeatureDistribution {

  val MaxBins = 100000
  val AvgBinValue = 5000
  val MaxTokenLowerLimit = 10
  val getBins = (sum: Summary, bins: Int) => bins
  // Todo: find out the right formula example:
  //  {
  //  To catch categoricals
  //    if (sum.max < MaxTokenLowerLimit) bins
  //    else math.min(math.max(bins, sum.sum / AvgBinValue), MaxBins).intValue()
  //  }

  implicit val semigroup: Semigroup[FeatureDistribution] = new Semigroup[FeatureDistribution] {
    override def plus(l: FeatureDistribution, r: FeatureDistribution) = l.reduce(r)
  }

  /**
   * Facilitates feature distribution retrieval from computed feature summaries
   *
   * @param featureKey feature key
   * @param summary feature summary
   * @param value optional processed sequence
   * @param bins number of histogram bins
   * @return feature distribution given the provided information
   */
  def apply(
    featureKey: FeatureKey,
    summary: Summary,
    value: Option[ProcessedSeq],
    bins: Int
  ): FeatureDistribution = {
    val (nullCount, (summaryInfo, distribution)): (Int, (Array[Double], Array[Double])) =
      value.map(seq => 0 -> histValues(seq, summary, bins, getBins))
        .getOrElse(1 -> (Array(summary.min, summary.max, summary.sum, summary.count) -> Array.fill(bins)(0.0)))

    FeatureDistribution(
      name = featureKey._1,
      key = featureKey._2,
      count = 1,
      nulls = nullCount,
      summaryInfo = summaryInfo,
      distribution = distribution)
  }

  /**
   * Function to put data into histogram of counts
   * @param values values to bin
   * @param sum summary info for feature (max and min)
   * @param bins number of bins to produce
   * @param getBins
   * @return the bin information and the binned counts
   */
  // TODO avoid wrapping and unwrapping??
  private def histValues(
    values: ProcessedSeq,
    sum: Summary,
    bins: Int,
    getBins: (Summary, Int) => Int
  ): (Array[Double], Array[Double]) = {
    values match {
      case Left(seq) => {
        val numBins = getBins(sum, bins)

        // Todo: creating too many hashers may cause problem, efficiency, garbage collection etc
        val hasher: HashingTF = new HashingTF(numFeatures = numBins)
          .setBinary(false)
          .setHashAlgorithm(HashAlgorithm.MurMur3.toString.toLowerCase)
        Array(sum.min, sum.max, sum.sum, sum.count) -> hasher.transform(seq).toArray
      }
      case Right(seq) => // TODO use kernel fit instead of histogram
        if (sum == Summary.empty) {
          Array(sum.min, sum.max) -> seq.toArray // the seq will always be empty in this case
        } else if (sum.min < sum.max) {
          val step = (sum.max - sum.min) / (bins - 2.0) // total number of bins includes one for edge and one for other
          val splits = (0 until bins).map(b => sum.min + step * b).toArray
          val binned = seq.map { v =>
            NumericBucketizer.bucketize(
              splits = splits, trackNulls = false, trackInvalid = true,
              splitInclusion = Inclusion.Left, input = Option(v)
            ).toArray
          }
          val hist = binned.fold(new Array[Double](bins))(_ + _)
          splits -> hist
        } else {
          val same = seq.map(v => if (v == sum.max) 1.0 else 0.0).sum
          val other = seq.map(v => if (v != sum.max) 1.0 else 0.0).sum
          Array(sum.min, sum.max, sum.sum, sum.count) -> Array(same, other)
        }
    }
  }
}
