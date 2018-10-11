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

import com.salesforce.op.OpParams
import com.salesforce.op.features.types._
import com.salesforce.op.features.{OPFeature, TransientFeature}
import com.salesforce.op.filters.FeatureDistribution._
import com.salesforce.op.filters.Summary._
import com.salesforce.op.readers.{DataFrameFieldNames, Reader}
import com.salesforce.op.stages.impl.feature.TimePeriod
import com.salesforce.op.stages.impl.preparators.CorrelationType
import com.salesforce.op.utils.spark.RichRow._
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.Tuple2Semigroup
import org.apache.spark.mllib.linalg.{Matrix, Vector}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.util.ClosureUtils
import org.slf4j.LoggerFactory

import scala.math.{abs, min}
import scala.util.Failure

/**
 * Specialized stage that will load up data and compute distributions and empty counts on raw features.
 * This information is then used to compute which raw features should be excluded from the workflow DAG
 * Note: Currently, raw features that aren't explicitly blacklisted, but are not used because they are inputs to
 * explicitly blacklisted features are not present as raw features in the model, nor in ModelInsights. However, they
 * are accessible from an OpWorkflowModel via getRawFeatureDistributions().
 *
 * @param trainingReader                reader to get the training data
 * @param scoreReader                   reader to get the scoring data for comparison (optional - if not present will
 *                                      exclude based on
 *                                      training data features only)
 * @param bins                          number of bins to use in computing feature distributions
 *                                      (histograms for numerics, hashes for strings)
 * @param minFill                       minimum fill rate a feature must have in the training dataset and
 *                                      scoring dataset to be kept
 * @param maxFillDifference             maximum acceptable fill rate difference between training
 *                                      and scoring data to be kept
 * @param maxFillRatioDiff              maximum acceptable fill ratio between training and scoring (larger / smaller)
 * @param maxJSDivergence               maximum Jensen-Shannon divergence between training
 *                                      and scoring distributions to be kept
 * @param maxCorrelation                maximum absolute correlation allowed between
 *                                      raw predictor null indicator and label
 * @param correlationType               type of correlation metric to use
 * @param jsDivergenceProtectedFeatures features that are protected from removal by JS divergence check
 * @param protectedFeatures             features that are protected from removal
 * @param textBinsFormula               formula to compute the text features bin size.
 *                                      Input arguments are [[Summary]] and number of bins to use in computing feature
 *                                      distributions (histograms for numerics, hashes for strings).
 *                                      Output is the bins for the text features.
 * @param timePeriod                    Time period used to apply circulate date transformation for date features, if
 *                                      not specified will use regular numeric feature transformation
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
  val maxCorrelation: Double,
  val correlationType: CorrelationType = CorrelationType.Pearson,
  val jsDivergenceProtectedFeatures: Set[String] = Set.empty,
  val protectedFeatures: Set[String] = Set.empty,
  val textBinsFormula: (Summary, Int) => Int = RawFeatureFilter.textBinsFormula,
  val timePeriod: Option[TimePeriod] = None
) extends Serializable {

  assert(bins > 1 && bins <= FeatureDistribution.MaxBins, s"Invalid bin size $bins," +
    s" bins must be between 1 and ${FeatureDistribution.MaxBins}")
  assert(minFill >= 0.0 && minFill <= 1.0, s"Invalid minFill size $minFill, minFill must be between 0 and 1")
  assert(maxFillDifference >= 0.0 && maxFillDifference <= 1.0, s"Invalid maxFillDifference size $maxFillDifference," +
    s" maxFillDifference must be between 0 and 1")
  assert(maxFillRatioDiff >= 0.0, s"Invalid maxFillRatioDiff size $maxFillRatioDiff," +
    s" maxFillRatioDiff must be greater than 0.0")
  assert(maxJSDivergence >= 0.0 && maxJSDivergence <= 1.0, s"Invalid maxJSDivergence size $maxJSDivergence," +
    s" maxJSDivergence must be between 0 and 1")

  ClosureUtils.checkSerializable(textBinsFormula) match {
    case Failure(e) => throw new AssertionError("The argument textBinsFormula must be serializable", e)
    case ok => ok
  }

  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  /**
   * Get binned counts of the feature distribution and empty count for each raw feature
   * @param data data frame to compute counts on
   * @param features list of raw, non-protected, features contained in the dataframe
   * @param allFeatureInfo existing feature info to use
   * @return a sequence of distribution summaries for each raw feature
   */
  private[op] def computeFeatureStats(
    data: DataFrame,
    features: Array[OPFeature],
    allFeatureInfo: Option[AllFeatureInformation] = None): AllFeatureInformation = {
    val (responses, predictors): (Array[TransientFeature], Array[TransientFeature]) = {
      val (allResponses, allPredictors) = features.partition(_.isResponse)
      val respOut = allResponses.map(TransientFeature(_)).flatMap {
        case f if f.getFeature().isSubtypeOf[OPNumeric[_]] =>
          log.info("Using numeric response: {}", f.name)
          Option(f)
        case f =>
          log.info("Not using non-numeric response in raw feature filter: {}", f.name)
          None
      }
      val predOut = allPredictors.map(TransientFeature(_))
      (respOut, predOut)
    }
    val preparedFeatures: RDD[PreparedFeatures] = data.rdd.map(PreparedFeatures(_, responses, predictors, timePeriod))

    implicit val sgTuple2Maps = new Tuple2Semigroup[Map[FeatureKey, Summary], Map[FeatureKey, Summary]]()
    // Have to use the training summaries do process scoring for comparison
    val (responseSummaries, predictorSummaries): (Map[FeatureKey, Summary], Map[FeatureKey, Summary]) =
      allFeatureInfo.map(info => info.responseSummaries -> info.predictorSummaries)
        .getOrElse(preparedFeatures.map(_.summaries).reduce(_ + _))
    val (responseSummariesArr, predictorSummariesArr): (Array[(FeatureKey, Summary)], Array[(FeatureKey, Summary)]) =
      (responseSummaries.toArray, predictorSummaries.toArray)

    implicit val sgTuple2Feats = new Tuple2Semigroup[Array[FeatureDistribution], Array[FeatureDistribution]]()
    val (responseDistributions, predictorDistributions): (Array[FeatureDistribution], Array[FeatureDistribution]) =
      preparedFeatures
        .map(_.getFeatureDistributions(
          responseSummaries = responseSummariesArr,
          predictorSummaries = predictorSummariesArr,
          bins = bins,
          textBinsFormula = textBinsFormula
        )).reduce(_ + _)
    val correlationInfo: Map[FeatureKey, Map[FeatureKey, Double]] =
      allFeatureInfo.map(_.correlationInfo).getOrElse {
        val responseKeys: Array[FeatureKey] = responseSummariesArr.map(_._1)
        val predictorKeys: Array[FeatureKey] = predictorSummariesArr.map(_._1)
        val corrRDD: RDD[Vector] = preparedFeatures.map(_.getNullLabelLeakageVector(responseKeys, predictorKeys))
        val corrMatrix: Matrix = Statistics.corr(corrRDD, correlationType.sparkName)

        responseKeys.zipWithIndex.map { case (responseKey, i) =>
          responseKey -> predictorKeys.zipWithIndex.map { case (predictorKey, j) =>
            predictorKey -> min(abs(corrMatrix(i, j + responseKeys.length)), 1.0)
          }.toMap
        }.toMap
      }

    AllFeatureInformation(
      responseSummaries = responseSummaries,
      responseDistributions = responseDistributions,
      predictorSummaries = predictorSummaries,
      predictorDistributions = predictorDistributions,
      correlationInfo = correlationInfo)
  }

  /**
   * Take in the distribution summaries for datasets (scoring summary may be empty) and determine which
   * features should be dropped (including maps with all keys dropped) and which map keys need to be dropped
   *
   * @param trainingDistribs summary of distributions for training data features
   * @param scoringDistribs  summary of distributions for scoring data features (may be an empty seq)
   * @param correlationInfo  info needed to determine feature to drop based on null label-leakage correlation
   * @return a list of feature names that should be dropped and a map of map keys that should be dropped
   *         Map(featureName -> key)
   */
  private[op] def getFeaturesToExclude(
    trainingDistribs: Seq[FeatureDistribution],
    scoringDistribs: Seq[FeatureDistribution],
    correlationInfo: Map[FeatureKey, Map[FeatureKey, Double]]
  ): (Seq[String], Map[String, Set[String]]) = {

    def logExcluded(excluded: Seq[Boolean], message: String): Unit = {
      val featuresDropped = trainingDistribs.zip(excluded)
        .collect{ case (f, d) if d => s"${f.name} ${f.key.getOrElse("")}" }
      log.info(s"$message: ${featuresDropped.mkString(", ")}")
    }

    val featureSize = trainingDistribs.length

    val trainingUnfilled = trainingDistribs.map(_.fillRate() < minFill)
    logExcluded(trainingUnfilled, s"Features excluded because training fill rate did not meet min required ($minFill)")

    val trainingNullLabelLeakers = {
      if (correlationInfo.isEmpty) Seq.fill(featureSize)(false)
      else {
        val absoluteCorrs = correlationInfo.map(_._2)
        for {distrib <- trainingDistribs} yield {
          // Only filter if feature absolute null-label leakage correlation is greater than allowed correlation
          val nullLabelLeakerIndicators = absoluteCorrs.map(_.get(distrib.featureKey).exists(_ > maxCorrelation))
          nullLabelLeakerIndicators.exists(identity)
        }
      }
    }
    logExcluded(
      trainingNullLabelLeakers,
      s"Features excluded because null indicator correlation (absolute) exceeded max allowed ($maxCorrelation)")

    val scoringUnfilled =
      if (scoringDistribs.nonEmpty) {
        assert(scoringDistribs.length == featureSize, "scoring and training features must match")
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
        val kl = combined.map { case (t, s) =>
          !jsDivergenceProtectedFeatures.contains(t.name) && t.jsDivergence(s) > maxJSDivergence
        }
        logExcluded(kl, s"Features excluded because JS Divergence exceeded max allowed ($maxJSDivergence)")
        val mf = combined.map { case (t, s) => t.relativeFillRate(s) > maxFillDifference }
        logExcluded(mf, s"Features excluded because fill rate difference exceeded max allowed ($maxFillDifference)")
        val mfr = combined.map { case (t, s) => t.relativeFillRatio(s) > maxFillRatioDiff }
        logExcluded(mfr, s"Features excluded because fill ratio difference exceeded max allowed ($maxFillRatioDiff)")
        kl.zip(mf).zip(mfr).map{ case ((a, b), c) => a || b || c }
      } else {
        Seq.fill(featureSize)(false)
      }

    val allExcludeReasons = trainingUnfilled.zip(scoringUnfilled).zip(distribMismatches).zip(trainingNullLabelLeakers)
      .map{ case (((t, s), d), n) => t || s || d || n }

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
   *
   * @param rawFeatures raw features used in the workflow
   * @param parameters  parameters used in the workflow
   * @param spark       spark instance
   * @return dataframe that has had bad features and bad map keys removed and a list of all features that should be
   *         dropped from the DAG
   */
  def generateFilteredRaw(rawFeatures: Array[OPFeature], parameters: OpParams)
    (implicit spark: SparkSession): FilteredRawData = {

    val trainData = trainingReader.generateDataFrame(rawFeatures, parameters).persist()
    log.info("Loaded training data")
    assert(trainData.count() > 0, "RawFeatureFilter cannot work with empty training data")
    val trainingSummary = computeFeatureStats(trainData, rawFeatures)
    log.info("Computed summary stats for training features")
    if (log.isDebugEnabled) {
      log.debug(trainingSummary.responseDistributions.mkString("\n"))
      log.debug(trainingSummary.predictorDistributions.mkString("\n"))
    }

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
      val ss = computeFeatureStats(sd, rawFeatures, Some(trainingSummary))
      log.info("Computed summary stats for scoring features")
      if (log.isDebugEnabled) {
        log.debug(ss.responseDistributions.mkString("\n"))
        log.debug(ss.predictorDistributions.mkString("\n"))
      }
      ss
    }

    val (featuresToDropNames, mapKeysToDrop) = getFeaturesToExclude(
      trainingSummary.predictorDistributions.filterNot(d => protectedFeatures.contains(d.name)),
      scoringSummary.toSeq.flatMap(_.predictorDistributions.filterNot(d => protectedFeatures.contains(d.name))),
      trainingSummary.correlationInfo)
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

    FilteredRawData(cleanedData, featuresToDrop, mapKeysToDrop,
      trainingSummary.responseDistributions ++ trainingSummary.predictorDistributions)
  }
}


object RawFeatureFilter {

  /**
   * Default calculation for the hashing size for RFF (compare js distance) for text features
   *
   * @param summary summary info for feature (max, min, etc)
   * @param bins number of bins to use
   * @return
   */
  def textBinsFormula(summary: Summary, bins: Int): Int = {
    // TODO: find out the right formula. Example:
    //  val AvgBinValue = 5000
    //  val MaxTokenLowerLimit = 10
    //  // To catch categoricals
    //  if (max < MaxTokenLowerLimit) bins
    //  else math.min(math.max(bins, sum / AvgBinValue), MaxBins).toInt()
    bins
  }

}

/**
 * case class for the RFF filtered data and features to drop
 *
 * @param cleanedData          RFF cleaned data
 * @param featuresToDrop       raw features dropped by RFF
 * @param mapKeysToDrop        keys in map features dropped by RFF
 * @param featureDistributions the feature distributions calculated from the training data
 */
case class FilteredRawData
(
  cleanedData: DataFrame,
  featuresToDrop: Array[OPFeature],
  mapKeysToDrop: Map[String, Set[String]],
  featureDistributions: Seq[FeatureDistribution]
)
