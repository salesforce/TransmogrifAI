/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.filters

import com.salesforce.op.OpParams
import com.salesforce.op.features.types._
import com.salesforce.op.features.{OPFeature, TransientFeature}
import com.salesforce.op.filters.FeatureDistrib.ProcessedSeq
import com.salesforce.op.readers.{DataFrameFieldNames, Reader}
import com.salesforce.op.stages.impl.feature.{HashAlgorithm, Inclusion, NumericBucketizer, TextTokenizer}
import com.salesforce.op.utils.spark.RichRow._
import com.salesforce.op.utils.text.Language
import com.twitter.algebird.Monoid
import com.twitter.algebird.Semigroup
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.LoggerFactory

/**
 * Specialized stage that will load up data and compute distributions and empty counts on raw features.
 * This information is then used to compute which raw features should be excluded from the workflow DAG
 * @param trainingReader reader to get the training data
 * @param scoreReader reader to get the scoring data for comparison (optional - if not present will exclude based on
 *                    training data features only)
 * @param bins number of bins to use in computing feature distributions (histograms for numerics, hashes for strings)
 * @param minFill minimum fill rate a feature must have in the training dataset and scoring dataset to be kept
 * @param maxFillDifference maximum acceptable fill rate difference between training and scoring data to be kept
 * @param maxFillRatioDiff maximum acceptable fill ratio between training and scoring (larger / smaller)
 * @param maxJSDivergence maximum Jensen-Shannon divergence between training and scoring distributions to be kept
 * @param protectedFeatures features that are protected from removal
 * @tparam T datatype of the reader
 */
class RawFeatureFilter[T]
(
  val trainingReader: Reader[T],
  val scoreReader: Option[Reader[T]],
  val bins: Int,
  val minFill: Double,
  val maxFillDifference: Double,
  val maxFillRatioDiff: Double,
  val maxJSDivergence: Double,
  val protectedFeatures: Set[String] = Set.empty
) extends Serializable {

  assert(bins > 1 && bins <= FeatureDistrib.MaxBins, s"Invalid bin size $bins," +
    s" bins must be between 1 and ${FeatureDistrib.MaxBins}")
  assert(minFill >= 0.0 && minFill <= 1.0, s"Invalid minFill size $minFill, minFill must be between 0 and 1")
  assert(maxFillDifference >= 0.0 && maxFillDifference <= 1.0, s"Invalid maxFillDifference size $maxFillDifference," +
    s" maxFillDifference must be between 0 and 1")
  assert(maxFillRatioDiff >= 0.0, s"Invalid maxFillRatioDiff size $maxFillRatioDiff," +
    s" maxFillRatioDiff must be greater than 0.0")
  assert(maxJSDivergence >= 0.0 && maxJSDivergence <= 1.0, s"Invalid maxJSDivergence size $maxJSDivergence," +
    s" maxJSDivergence must be between 0 and 1")

  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  private val hasher: HashingTF = new HashingTF(numFeatures = bins)
    .setBinary(false)
    .setHashAlgorithm(HashAlgorithm.MurMur3.toString.toLowerCase)

  private def tokenize(s: String) = TextTokenizer.Analyzer.analyze(s, Language.Unknown)

  /**
   * Turn features into a sequence that will have stats computed on it based on the type of the feature
   * @param value feature value
   * @tparam T type of the feature
   * @return a tuple containing whether the feature was empty and a sequence of either doubles or strings
   */
  private def prepareFeatures[T <: FeatureType](value: T): (Boolean, ProcessedSeq) = {
    value match {
      case v: Text => v.isEmpty -> Left(v.value.map(tokenize).getOrElse(Seq.empty)) // TODO are empty strings == nulls
      case v: OPNumeric[_] => v.isEmpty -> Right(v.toDouble.toSeq)
      case v: OPVector => v.isEmpty -> Right(v.value.toArray.toSeq)
      case v: Geolocation => v.isEmpty -> Right(v.value)
      case v: TextList => v.isEmpty -> Left(v.value)
      case v: DateList => v.isEmpty -> Right(v.value.map(_.toDouble))
      case v: MultiPickList => v.isEmpty -> Left(v.value.toSeq)
      case _ => throw new RuntimeException(s"Feature type $value is not supported in RawFeatureFilter")
    }
  }


  /**
   * Turn map features into a map of sequences that will have stats computed on it based on the type of the feature
   * @param value feature value
   * @tparam T type of the map feature
   * @return a map from the keys to a sequence of either doubles or strings
   */
  private def prepareMapFeatures[T <: FeatureType](value: T): Map[String, ProcessedSeq] = {
    value match {
      case v: MultiPickListMap => v.value.map{ case (k, e) => k -> Left(e.toSeq) }
      case v: GeolocationMap => v.value.map{ case (k, e) => k -> Right(e) }
      case v: OPMap[_] => v.value.map { case (k, e) => e match {
        case d: Double => k -> Right(Seq(d))
        case s: String => k -> Left(tokenize(s))
        case l: Long => k -> Right(Seq(l.toDouble))
        case b: Boolean => k -> Right(Seq(if (b) 1.0 else 0.0))
      }}
      case _ => throw new RuntimeException(s"Feature type $value is not supported in RawFeatureFilter")
    }
  }

  /**
   * Get binned counts of the feature distribution and empty count for each raw feature
   * @param data data frame to compute counts on
   * @param rawFeatures list of raw features contained in the dataframe
   * @return a sequence of distribution summaries for each raw feature
   */
  // TODO do these computations on a per label basis??
  private[op] def computeFeatureStats(data: DataFrame, rawFeatures: Array[OPFeature],
    featureSummaries: Option[AllFeatureInformation] = None): AllFeatureInformation = {
    val (mapTranFeatures, tranFeatures) = rawFeatures
      .map(f => TransientFeature(f) -> FeatureTypeSparkConverter()(f.wtt))
      .partition(_._1.getFeature().isSubtypeOf[OPMap[_]])

    val preparedFeatures = data.rdd.map{ row =>
      tranFeatures.map(f => prepareFeatures(row.getFeatureType(f._1)(f._2))) ->
        mapTranFeatures.map(mf => prepareMapFeatures(row.getFeatureType(mf._1)(mf._2)))
    }

    val (summaryFeatures, summaryMapFeatures) = // Have to use the training summaries do process scoring for comparison
      featureSummaries.map{ fs => fs.featureSummaries -> fs.mapFeatureSummaries }.getOrElse{
        preparedFeatures.map { case (features, mapFeatures) =>
          features.map(f => Summary(f._2)) -> mapFeatures.map(mf => mf.map { case (k, v) => k -> Summary(v) })
        }.reduce(_ + _)
      }

    val featureDistrib = preparedFeatures
      .map{ case (features, mapFeatures) =>
        FeatureDistrib.getDistributions(tranFeatures.map(_._1), features, summaryFeatures, bins, hasher) ++
          FeatureDistrib.getMapDistributions(mapTranFeatures.map(_._1), mapFeatures, summaryMapFeatures, bins, hasher) }
      .reduce(_ + _)

    AllFeatureInformation(summaryFeatures, summaryMapFeatures, featureDistrib)
  }

  /**
   * Take in the distribution summaries for datasets (scoring summary may be empty) and determine which
   * features should be dropped (including maps with all keys dropped) and which map keys need to be dropped
   * @param trainingDistribs summary of distributions for training data features
   * @param scoringDistribs summary of distributions for scoring data features (may be an empty seq)
   * @return a list of feature names that should be dropped and a map of map keys that should be dropped
   *         Map(featureName -> key)
   */
  private[op] def getFeaturesToExclude(
    trainingDistribs: Seq[FeatureDistrib],
    scoringDistribs: Seq[FeatureDistrib]
  ): (Seq[String], Map[String, Set[String]]) = {

    def logExcluded(excluded: Seq[Boolean], message: String): Unit = {
      val featuresDropped = trainingDistribs.zip(excluded)
        .collect{ case (f, d) if d => s"${f.name} ${f.key.getOrElse("")}" }
      log.info(s"$message: ${featuresDropped.mkString(", ")}")
    }

    val featureSize = trainingDistribs.size

    val trainingUnfilled = trainingDistribs.map(_.fillRate() < minFill)
    logExcluded(trainingUnfilled, s"Features excluded because training fill rate did not meet min required ($minFill)")

    val scoringUnfilled =
      if (scoringDistribs.nonEmpty) {
        assert(scoringDistribs.length == trainingDistribs.length, "scoring and training features must match")
        val su = scoringDistribs.map(_.fillRate() < minFill)
        logExcluded(su, s"Features excluded because scoring fill rate did not meet min required ($minFill)")
        su
      } else {
        Seq.fill(featureSize)(false)
      }

    val distribMismatches =
      if (scoringDistribs.nonEmpty) {
        val combined = trainingDistribs.zip(scoringDistribs)
        log.info(combined.map { case (t, s) => s"\n$t\n$s\nTrain Fill=${t.fillRate()}, Score Fill=${s.fillRate()}, " +
          s"JS Divergence=${t.jsDivergence(s)}, Fill Rate Difference=${t.relativeFillRate(s)}, " +
          s"Fill Ratio Difference=${t.relativeFillRatio(s)}" }.mkString("\n"))
        val kl = combined.map { case (t, s) => t.jsDivergence(s) > maxJSDivergence }
        logExcluded(kl, s"Features excluded because JS Divergence exceeded max allowed ($maxJSDivergence)")
        val mf = combined.map { case (t, s) => t.relativeFillRate(s) > maxFillDifference }
        logExcluded(mf, s"Features excluded because fill rate difference exceeded max allowed ($maxFillDifference)")
        val mfr = combined.map { case (t, s) => t.relativeFillRatio(s) > maxFillRatioDiff }
        logExcluded(mfr, s"Features excluded because fill ratio difference exceeded max allowed ($maxFillRatioDiff)")
        kl.zip(mf).zip(mfr).map{ case ((a, b), c) => a || b || c }
      } else {
        Seq.fill(featureSize)(false)
      }

    val allExcludeReasons = trainingUnfilled.zip(scoringUnfilled).zip(distribMismatches)
      .map{ case ((t, s), d) => t || s || d }

    val (toDrop, toKeep) = trainingDistribs.zip(allExcludeReasons).partition(_._2)

    val toDropFeatures = toDrop.map(_._1).groupBy(_.name)
    val toKeepFeatures = toKeep.map(_._1).groupBy(_.name)
    val mapKeys = toKeepFeatures.keySet.intersect(toDropFeatures.keySet)
    val toDropNames = toDropFeatures.collect{ case (k, _) if !mapKeys.contains(k) => k }.toSeq
    val toDropMapKeys = toDropFeatures.collect{ case (k, v) if mapKeys.contains(k) => k -> v.flatMap(_.key).toSet }
    toDropNames -> toDropMapKeys
  }

  /**
   * Function that gets raw features and params used in workflow. Will use this information along with readers for this
   * stage to determine which features should be dropped from the workflow
   * @param rawFeatures raw features used in the workflow
   * @param parameters parameters used in the workflow
   * @param spark spark instance
   * @return dataframe that has had bad features and bad map keys removed and a list of all features that should be
   *         dropped from the DAG
   */
  // TODO return distribution information to attach to features that are kept
  def generateFilteredRaw(rawFeatures: Array[OPFeature], parameters: OpParams)
    (implicit spark: SparkSession): (DataFrame, Array[OPFeature]) = {

    val (_, predictorFeatures) = rawFeatures.partition(f => f.isResponse || protectedFeatures.contains(f.name) )

    val trainData = trainingReader.generateDataFrame(rawFeatures, parameters).persist()
    log.info("Loaded training data")
    assert(trainData.count() > 0, "RawFeatureFilter cannot work with empty training data")
    val trainingSummary = computeFeatureStats(trainData, predictorFeatures) // TODO also response summaries??
    log.info("Computed summary stats for training features")
    log.debug(trainingSummary.featureDistributions.mkString("\n"))

    val scoreData = scoreReader.flatMap{ s =>
      val sd = s.generateDataFrame(rawFeatures, parameters.switchReaderParams()).persist()
      log.info("Loaded scoring data")
      if (sd.count() > 0) Some(sd)
      else {
        log.warn("Scoring dataset was empty. Only training data checks will be used.")
        None
      }
    }

    val scoringSummary = scoreData.map{ sd =>
      val ss = computeFeatureStats(sd, predictorFeatures, Some(trainingSummary)) // TODO also response summaries??
      log.info("Computed summary stats for scoring features")
      log.debug(ss.featureDistributions.mkString("\n"))
      ss
    }

    val (featuresToDropNames, mapKeysToDrop) = getFeaturesToExclude(
      trainingSummary.featureDistributions,
      scoringSummary.toSeq.flatMap(_.featureDistributions)
    )
    val (featuresToDrop, featuresToKeep) = rawFeatures.partition(rf => featuresToDropNames.contains(rf.name))
    val featuresToKeepNames = Array(DataFrameFieldNames.KeyFieldName) ++ featuresToKeep.map(_.name)

    val featuresDropped = trainData.drop(featuresToDropNames: _*)
    val mapsCleaned = featuresDropped.rdd.map{ row =>
      val kept = featuresToKeepNames.map{ fn =>
        if (mapKeysToDrop.contains(fn)) {
          val map = row.getMapAny(fn)
          if (map != null) map.filterNot{ case (k, _) => mapKeysToDrop(fn).contains(k) } else map
        } else {
          row.getAny(fn)
        }
      }
      Row.fromSeq(kept)
    }

    val schema = StructType(featuresToKeepNames.map(featuresDropped.schema(_)))
    val cleanedData = spark.createDataFrame(mapsCleaned, schema).persist()
    trainData.unpersist()
    scoreData.map(_.unpersist())

    cleanedData -> featuresToDrop
  }
}

private[op] case class AllFeatureInformation
(
  featureSummaries: Array[Summary],
  mapFeatureSummaries: Array[Map[String, Summary]],
  featureDistributions: Array[FeatureDistrib]
)

/**
 * Class used to get summaries of prepped features so know how to bin it for distributions
 * @param min minimum value seen
 * @param max maximum value seen
 */
private[op] case class Summary(min: Double, max: Double)

private[op] case object Summary {

  val empty: Summary = Summary(Double.PositiveInfinity, Double.NegativeInfinity)

  implicit val monoid: Monoid[Summary] = new Monoid[Summary] {
    override def zero = empty
    override def plus(l: Summary, r: Summary) = Summary(math.min(l.min, r.min), math.max(l.max, r.max))
  }

  def apply(preppedFeature: ProcessedSeq): Summary = {
    preppedFeature match {
      case Left(v) => Summary(v.size, v.size)
      case Right(v) => monoid.sum(v.map(d => Summary(d, d)))
    }
  }
}


/**
 * Class containing summary information for a feature
 * @param name name of the feature
 * @param key map key associated with distribution (when the feature is a map)
 * @param count total count of feature seen
 * @param nulls number of empties seen in feature
 * @param distribution binned counts of feature values (hashed for strings, evently spaced bins for numerics)
 * @param summaryInfo either min and max of data (for text data) or splits used for bins for numeric data
 */
case class FeatureDistrib
(
  name: String,
  key: Option[String],
  count: Long,
  nulls: Long,
  distribution: Array[Double],
  summaryInfo: Array[Double]
) {

  /**
   * Check that feature distributions below to the same feature and key
   * @param fd distribution to compare to
   */
  def checkMatch(fd: FeatureDistrib): Unit =
    assert(name == fd.name && key == fd.key, "Name and key must match to compare or combine FeatureDistrib")

  /**
   * Get fill rate of feature
   * @return fraction of data that is non empty
   */
  def fillRate(): Double = if (count == 0L) 0.0 else (count - nulls) / count.toDouble

  /**
   * Combine feature distributions
   * @param fd other feature distribution (from the same feature)
   * @return summed distribution information
   */
  def reduce(fd: FeatureDistrib): FeatureDistrib = {
    checkMatch(fd)
    val combinedDist = distribution + fd.distribution
    // summary info can be empty or min max if hist is empty but should otherwise match so take the longest info
    val combinedSummary = if (summaryInfo.length > fd.summaryInfo.length) summaryInfo else fd.summaryInfo
    FeatureDistrib(name, key, count + fd.count, nulls + fd.nulls, combinedDist, combinedSummary)
  }

  /**
   * Ratio of fill rates between the two distributions symetric with larger value on the top
   * @param fd feature distribution to compare to
   * @return ratio of fill rates
   */
  def relativeFillRatio(fd: FeatureDistrib): Double = {
    checkMatch(fd)
    val (thisFill, thatFill) = (fillRate(), fd.fillRate())
    val (small, large) = if (thisFill < thatFill) (thisFill, thatFill) else (thatFill, thisFill)
    if (small == 0.0) Double.PositiveInfinity else large / small
  }

  /**
   * Absolute difference in empty rates
   * @param fd feature distribution to compare to
   * @return fill rate ratio with larger fill rate on the bottom
   */
  def relativeFillRate(fd: FeatureDistrib): Double = {
    checkMatch(fd)
    math.abs(fillRate() - fd.fillRate())
  }

  /**
   * Jensen-Shannon divergence from this distribution to the other distribution fed in
   * @param fd other feature distribution
   * @return the KL divergence
   */
  def jsDivergence(fd: FeatureDistrib): Double = {
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

private[op] case object FeatureDistrib {

  type ProcessedSeq = Either[Seq[String], Seq[Double]]

  val MaxBins = 100000

  implicit val semigroup: Semigroup[FeatureDistrib] = new Semigroup[FeatureDistrib] {
    override def plus(l: FeatureDistrib, r: FeatureDistrib) = l.reduce(r)
  }

  /**
   * Function to put data into histogram of counts
   * @param values values to bin
   * @param sum summary info for feature (max and min)
   * @param bins number of bins to produce
   * @param hasher hasing function to use for text
   * @return the bin information and the binned counts
   */
  // TODO avoid wrapping and unwrapping??
  private def histValues(
    values: ProcessedSeq,
    sum: Summary,
    bins: Int,
    hasher: HashingTF
  ): (Array[Double], Array[Double]) = {
    values match {
      case Left(seq) => Array(sum.min, sum.max) -> hasher.transform(seq).toArray // TODO use summary info to pick hashes
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
          Array(sum.min, sum.max) -> Array(same, other)
        }
    }
  }

  /**
   * Create the distributions for regular features
   * @param features list of transient features
   * @param values values of the features processed into a sequence of either doubles or strings with boolean
   *               indicating if original feature was empty
   * @param summary summary statistics about feature
   * @param bins number of bins to put numerics into
   * @param hasher hash function to use on strings
   * @return feature distribution for single feature value to be aggregated
   */
  def getDistributions(
    features: Array[TransientFeature],
    values: Array[(Boolean, ProcessedSeq)],
    summary: Array[Summary],
    bins: Int,
    hasher: HashingTF
  ): Array[FeatureDistrib] = {
    features.zip(values).zip(summary).map{
      case ((tf, (isNull, seq)), sum) =>
        val isNullCount = if (isNull) 1 else 0
        val (info, histogram) = histValues(seq, sum, bins, hasher)
        FeatureDistrib(tf.name, None, 1, isNullCount, histogram, info)
    }
  }

  /**
   * Create the distributions for map features
   * @param features list of transient map features
   * @param values values of the features processed into a map from key to sequence of either doubles or strings
   * @param summary map from key to summary statistics about feature
   * @param bins number of bins to put numerics into
   * @param hasher hash function to use on strings
   * @return feature distribution for single feature and key value to be aggregated
   */
  def getMapDistributions(
    features: Array[TransientFeature],
    values: Array[Map[String, ProcessedSeq]],
    summary: Array[Map[String, Summary]],
    bins: Int,
    hasher: HashingTF
  ): Array[FeatureDistrib] = {
    features.zip(values).zip(summary).flatMap {
      case ((tf, map), sum) => sum.map { case (key, seq) =>
        val isNullCount = if (map.contains(key)) 0 else 1
        val (info, histogram) = map.get(key)
          .map(seq => histValues(seq, sum(key), bins, hasher))
          .getOrElse(Array(sum(key).min, sum(key).max), Array.fill(bins)(0.0))
        FeatureDistrib(tf.name, Some(key), 1, isNullCount, histogram, info)
      }
    }
  }

}
