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
import com.salesforce.op.features.{FeatureDistributionType, OPFeature, TransientFeature}
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
 * are accessible from an OpWorkflowModel via getRawFeatureFilterResults().
 *
 * @param trainingReader                reader to get the training data
 * @param scoringReader                 reader to get the scoring data for comparison
 *                                      (optional - if not present will exclude based on training data features only)
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
 * @param minScoringRows                Minimum row threshold for scoring set comparisons to be used in checks. If
 *                                      the scoring set size is below this threshold, then only training data checks
 *                                      will be used
 * @tparam T datatype of the reader
 */
class RawFeatureFilter[T]
(
  val trainingReader: Reader[T],
  val scoringReader: Option[Reader[T]],
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
  val timePeriod: Option[TimePeriod] = None,
  val minScoringRows: Int = RawFeatureFilter.minScoringRowsDefault
) extends Serializable {

  require(bins > 1 && bins <= FeatureDistribution.MaxBins, s"Invalid bin size $bins," +
    s" bins must be between 1 and ${FeatureDistribution.MaxBins}")
  require(minFill >= 0.0 && minFill <= 1.0, s"Invalid minFill size $minFill, minFill must be between 0 and 1")
  require(maxFillDifference >= 0.0 && maxFillDifference <= 1.0, s"Invalid maxFillDifference size $maxFillDifference," +
    s" maxFillDifference must be between 0 and 1")
  require(maxFillRatioDiff >= 0.0, s"Invalid maxFillRatioDiff size $maxFillRatioDiff," +
    s" maxFillRatioDiff must be greater than 0.0")
  require(maxJSDivergence >= 0.0 && maxJSDivergence <= 1.0, s"Invalid maxJSDivergence size $maxJSDivergence," +
    s" maxJSDivergence must be between 0 and 1")
  require(minScoringRows >= 0, s"minRowsForScoringSet must be >= 0, but was set to $minScoringRows")

  ClosureUtils.checkSerializable(textBinsFormula) match {
    case Failure(e) => throw new IllegalArgumentException("The argument textBinsFormula must be serializable", e)
    case ok => ok
  }

  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  /**
   * Get binned counts of the feature distribution and empty count for each raw feature
   * statistics on the training and scoring data. It does two map reduce operations, the first to produce a Summary
   * of each feature, the second to produce a binned histogram (Distribution) for each feature based on the Summary.
   * @param data                    data frame to compute counts on
   * @param features                list of raw, non-protected, features contained in the dataframe
   * @param featureDistributionType feature distribution type: training or scoring
   * @param allFeatureInfo          existing feature info to use
   * @return a sequence of distribution summaries for each raw feature
   */
  private[op] def computeFeatureStats(
    data: DataFrame,
    features: Array[OPFeature],
    featureDistributionType: FeatureDistributionType,
    allFeatureInfo: Option[AllFeatureInformation] = None
  ): AllFeatureInformation = {
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
    // process all features based on raw type so that they can be summerized as either text or numeric
    val preparedFeatures: RDD[PreparedFeatures] = data.rdd.map(PreparedFeatures(_, responses, predictors, timePeriod))

    implicit val sgTuple2Maps = new Tuple2Semigroup[Map[FeatureKey, Summary], Map[FeatureKey, Summary]]()
    // Have to use the training summaries do process scoring for comparison
    val (responseSummaries, predictorSummaries): (Map[FeatureKey, Summary], Map[FeatureKey, Summary]) =
      allFeatureInfo.map { info =>
        info.responseSummaries -> info.predictorSummaries
      }.getOrElse(preparedFeatures.map(_.summaries).reduce(_ + _))
    val (responseSummariesArr, predictorSummariesArr): (Array[(FeatureKey, Summary)], Array[(FeatureKey, Summary)]) =
      (responseSummaries.toArray, predictorSummaries.toArray)

    implicit val sgTuple2Feats = new Tuple2Semigroup[Array[FeatureDistribution], Array[FeatureDistribution]]()
    val (responseDistributions, predictorDistributions): (Array[FeatureDistribution], Array[FeatureDistribution]) =
      preparedFeatures
        .map(_.getFeatureDistributions(
          responseSummaries = responseSummariesArr,
          predictorSummaries = predictorSummariesArr,
          bins = bins,
          textBinsFormula = textBinsFormula,
          featureDistributionType
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
      correlationInfo = correlationInfo
    )
  }

  /**
   * Take in the distribution summaries for datasets (scoring summary may be empty) and return metrics that
   * are used by raw feature filter
   *
   * @param trainingDistribs summary of distributions for training data features
   * @param scoringDistribs  summary of distributions for scoring data features (may be an empty seq)
   * @param correlationInfo  info needed to determine feature to drop based on null label-leakage correlation
   * @return metrics computed by raw feature filter tests for each feature
   */
  private[op] def getRawFeatureFilterMetrics(
    trainingDistribs: Seq[FeatureDistribution],
    scoringDistribs: Seq[FeatureDistribution],
    correlationInfo: Map[FeatureKey, Map[FeatureKey, Double]]
  ): Seq[RawFeatureFilterMetrics] = {

    val featureSize: Int = trainingDistribs.length

    val trainingFillRates: Seq[Double] = trainingDistribs.map(_.fillRate())
    val trainingCardSizes: Seq[Option[Int]] = trainingDistribs.map(_.cardSize)
    val trainingNullLabelAbsoluteCorrs: Seq[Option[Double]] =
      if (correlationInfo.isEmpty) Seq.fill(featureSize)(None)
      else {
        val absoluteCorrs =
          for {
            distrib <- trainingDistribs
          } yield correlationInfo.values.map(_.get(distrib.featureKey))
        absoluteCorrs.flatten
      }

    // combines metrics computed for each feature into RawFeatureMetrics class
    def combineRawFeatureFilterMetrics(
      traininingDistribs: Seq[FeatureDistribution],
      trainingFillRates: Seq[Double],
      trainingNullLabelAbsoluteCorrs: Seq[Option[Double]],
      trainingCardSizes: Seq[Option[Int]],
      scoringFillRates: Seq[Option[Double]],
      jsDivergences: Seq[Option[Double]],
      fillRateDiffs: Seq[Option[Double]],
      fillRatioDiffs: Seq[Option[Double]]
    ): Seq[RawFeatureFilterMetrics] = {

      trainingDistribs.map(dist => dist.name -> dist.key)
        .zip(trainingFillRates)
        .zip(trainingNullLabelAbsoluteCorrs)
        .zip(scoringFillRates)
        .zip(jsDivergences)
        .zip(fillRateDiffs)
        .zip(fillRatioDiffs)
        .zip(trainingCardSizes)
        .map {
          case ((((((((name, key), trainingFillRate), trainingNullLabelAbsoluteCorr),
          scoringFillRate), jsDivergence), fillRateDiff), fillRatioDiff),
          trainingCardSize) =>
            RawFeatureFilterMetrics(
              name, key, trainingFillRate, trainingNullLabelAbsoluteCorr,
              scoringFillRate, jsDivergence, fillRateDiff, fillRatioDiff,
              trainingCardSize)
        }
    }

    if (scoringDistribs.isEmpty) {

      val scoringFillRates = Seq.fill(featureSize)(None)
      val jsDivergences = Seq.fill(featureSize)(None)
      val fillRateDiffs = Seq.fill(featureSize)(None)
      val fillRatioDiffs = Seq.fill(featureSize)(None)

      val rawFeatureFilterMetrics = combineRawFeatureFilterMetrics(
        trainingDistribs, trainingFillRates, trainingNullLabelAbsoluteCorrs,
        trainingCardSizes, scoringFillRates, jsDivergences, fillRateDiffs, fillRatioDiffs
      )
      rawFeatureFilterMetrics

    } else {

      require(scoringDistribs.length == featureSize, "scoring and training features must match")

      val scoringFillRates = scoringDistribs.map(distrib => Option(distrib.fillRate()))

      val combined = trainingDistribs.zip(scoringDistribs)
      val jsDivergences = combined.map { case (t, s) => Option(t.jsDivergence(s)) }
      val fillRateDiffs = combined.map { case (t, s) => Option(t.relativeFillRate(s)) }
      val fillRatioDiffs = combined.map { case (t, s) => Option(t.relativeFillRatio(s)) }

      val rawFeatureFilterMetrics = combineRawFeatureFilterMetrics(
        trainingDistribs, trainingFillRates, trainingNullLabelAbsoluteCorrs,
        trainingCardSizes, scoringFillRates, jsDivergences,
        fillRateDiffs, fillRatioDiffs
      )

      log.info(combined.zip(rawFeatureFilterMetrics).map {
        case ((t, s), m) => s"\n$t\n$s\nTrain Fill=${m.trainingFillRate}, Score Fill=${m.scoringFillRate}, " +
        s"JS Divergence=${m.jsDivergence}, Fill Rate Difference=${m.fillRateDiff}, " +
        s"Fill Ratio Difference=${m.fillRatioDiff}"
      }.mkString("\n"))

      rawFeatureFilterMetrics
    }
  }

  /**
   * Take in the distributions and metrics for datasets (scoring may be empty) and return outcomes of raw feature filter
   * tests for each feature
   *
   * @param trainingDistribs        summary of distributions for training data features
   * @param scoringDistribs         summary of distributions for scoring data features (may be an empty seq)
   * @param rawFeatureFilterMetrics metrics used in raw feature filters
   * @return a list of outcomes of raw feature filter tests for each feature
   */
  private[op] def getRawFeatureFilterExclusionReasons(
    trainingDistribs: Seq[FeatureDistribution],
    scoringDistribs: Seq[FeatureDistribution],
    rawFeatureFilterMetrics: Seq[RawFeatureFilterMetrics]
  ): Seq[ExclusionReasons] = {

    def logExcluded(excluded: Seq[Boolean], message: String): Unit = {
      val featuresDropped = trainingDistribs.zip(excluded)
        .collect { case (f, d) if d => s"${f.name} ${f.key.getOrElse("")}" }
      log.info(s"$message: ${featuresDropped.mkString(", ")}")
    }

    val featureSize: Int = trainingDistribs.length

    val trainingUnfilledStates: Seq[Boolean] = rawFeatureFilterMetrics.map(_.trainingFillRate < minFill)

    logExcluded(
      excluded = trainingUnfilledStates,
      message = s"Features excluded because training fill rate did not meet min required ($minFill)"
    )

    val trainingNullLabelLeakers: Seq[Boolean] = rawFeatureFilterMetrics.map(_.trainingNullLabelAbsoluteCorr).map {
      case Some(corr) => corr > maxCorrelation
      case None => false
    }

    logExcluded(
      excluded = trainingNullLabelLeakers,
      message = "Features excluded because null indicator correlation (absolute) " +
        s"exceeded max allowed ($maxCorrelation)"
    )

    // combines exclusion reasons computed for each feature into ExclusionReasons class
    def combineExclusionReasons(
      traininingDistribs: Seq[FeatureDistribution],
      trainingUnfilledStates: Seq[Boolean],
      trainingNullLabelLeakers: Seq[Boolean],
      scoringUnfilledStates: Seq[Boolean],
      jsDivergenceMismatches: Seq[Boolean],
      fillRateDiffMismatches: Seq[Boolean],
      fillRatioDiffMismatches: Seq[Boolean]
    ): Seq[ExclusionReasons] = {

      trainingDistribs.map(dist => dist.name-> dist.key)
        .zip(trainingUnfilledStates)
        .zip(trainingNullLabelLeakers)
        .zip(scoringUnfilledStates)
        .zip(jsDivergenceMismatches)
        .zip(fillRateDiffMismatches)
        .zip(fillRatioDiffMismatches)
        .map {
          case (((((((name, key), trainingUnfilledState), trainingNullLabelLeaker),
          scoringUnfilledState), jsDivergenceMismatch),
          fillRateDiffMismatch), fillRatioDiffMismatch) =>
            ExclusionReasons(
              name,
              key,
              trainingUnfilledState,
              trainingNullLabelLeaker,
              scoringUnfilledState,
              jsDivergenceMismatch,
              fillRateDiffMismatch,
              fillRatioDiffMismatch,
              excluded = List(
                trainingUnfilledState,
                trainingNullLabelLeaker,
                scoringUnfilledState,
                jsDivergenceMismatch,
                fillRateDiffMismatch,
                fillRatioDiffMismatch
              ).exists(identity)
            )
        }
    }

    if (scoringDistribs.isEmpty) {

      val scoringUnfilledStates = Seq.fill(featureSize)(false)
      val jsDivergenceMismatches = Seq.fill(featureSize)(false)
      val fillRateDiffMismatches = Seq.fill(featureSize)(false)
      val fillRatioDiffMismatches = Seq.fill(featureSize)(false)

      val exclusionReasons: Seq[ExclusionReasons] = combineExclusionReasons(
        trainingDistribs, trainingUnfilledStates, trainingNullLabelLeakers,
        scoringUnfilledStates, jsDivergenceMismatches, fillRateDiffMismatches, fillRatioDiffMismatches
      )
      exclusionReasons

    } else {

      val scoringUnfilledStates: Seq[Boolean] = rawFeatureFilterMetrics.map(_.scoringFillRate).map {
        case Some(scoringFillRate) => scoringFillRate < minFill
        case None => false
      }

      val jsDivergenceMismatches = rawFeatureFilterMetrics.map { m =>
        !jsDivergenceProtectedFeatures.contains(m.name) &&
          (m.jsDivergence match {
            case Some(jsDivergence) => jsDivergence > maxJSDivergence
            case None => false
          })
      }

      val fillRateDiffMismatches = rawFeatureFilterMetrics.map(_.fillRateDiff).map {
        case Some(fillRateDiff) => fillRateDiff > maxFillDifference
        case None => false
      }

      val fillRatioDiffMismatches = rawFeatureFilterMetrics.map(_.fillRatioDiff).map {
        case Some(fillRatioDiff) => fillRatioDiff > maxFillRatioDiff
        case None => false
      }

      logExcluded(scoringUnfilledStates,
        s"Features excluded because scoring fill rate did not meet min required ($minFill)")
      logExcluded(jsDivergenceMismatches,
        s"Features excluded because JS Divergence exceeded max allowed ($maxJSDivergence)")
      logExcluded(fillRateDiffMismatches,
        s"Features excluded because fill rate difference exceeded max allowed ($maxFillDifference)")
      logExcluded(fillRatioDiffMismatches,
        s"Features excluded because fill ratio difference exceeded max allowed ($maxFillRatioDiff)")

      val exclusionReasons: Seq[ExclusionReasons] = combineExclusionReasons(
        trainingDistribs, trainingUnfilledStates, trainingNullLabelLeakers,
        scoringUnfilledStates, jsDivergenceMismatches, fillRateDiffMismatches, fillRatioDiffMismatches
      )
      exclusionReasons
    }
  }

  /**
   * Take in the distribution summaries for datasets (scoring summary may be empty) and determine which
   * features should be dropped (including maps with all keys dropped), which map keys need to be dropped, and
   * why those features are dropped
   *
   * @param trainingDistribs summary of distributions for training data features
   * @param scoringDistribs  summary of distributions for scoring data features (may be an empty seq)
   * @param correlationInfo  info needed to determine feature to drop based on null label-leakage correlation
   * @return a list of outcomes of raw feature filter tests for each feature, a list of feature names that
   *         should be dropped and a map of map keys that should be dropped Map(featureName -> key)
   */
  private[op] def getFeaturesToExclude(
    trainingDistribs: Seq[FeatureDistribution],
    scoringDistribs: Seq[FeatureDistribution],
    correlationInfo: Map[FeatureKey, Map[FeatureKey, Double]]
  ): (Seq[RawFeatureFilterMetrics], Seq[ExclusionReasons], Seq[String], Map[String, Set[String]]) = {

    val rawFeatureFilterMetrics = getRawFeatureFilterMetrics(
      trainingDistribs = trainingDistribs,
      scoringDistribs = scoringDistribs,
      correlationInfo = correlationInfo
    )

    val exclusionReasons = getRawFeatureFilterExclusionReasons(
      trainingDistribs = trainingDistribs,
      scoringDistribs = scoringDistribs,
      rawFeatureFilterMetrics = rawFeatureFilterMetrics
    )

    val excludedFeatures = exclusionReasons.map(_.excluded)

    val (toDrop, toKeep) = trainingDistribs.zip(excludedFeatures).partition(_._2)

    val toDropFeatures = toDrop.map(_._1).groupBy(_.name)
    val toKeepFeatures = toKeep.map(_._1).groupBy(_.name)
    val mapKeys = toKeepFeatures.keySet.intersect(toDropFeatures.keySet)
    val toDropNames = toDropFeatures.collect { case (k, _) if !mapKeys.contains(k) => k }.toSeq
    val toDropMapKeys = toDropFeatures.collect { case (k, v) if mapKeys.contains(k) => k -> v.flatMap(_.key).toSet }

    (rawFeatureFilterMetrics, exclusionReasons, toDropNames, toDropMapKeys)
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
    require(trainData.count() > 0, "RawFeatureFilter cannot work with empty training data")
    val trainingSummary = computeFeatureStats(trainData, rawFeatures, FeatureDistributionType.Training)
    log.info("Computed summary stats for training features")
    if (log.isDebugEnabled) {
      log.debug(trainingSummary.responseDistributions.mkString("\n"))
      log.debug(trainingSummary.predictorDistributions.mkString("\n"))
    }

    val scoreData = scoringReader.flatMap { s =>
      val sd = s.generateDataFrame(rawFeatures, parameters.switchReaderParams()).persist()
      log.info("Loaded scoring data")
      val scoringDataCount = sd.count()
      if (scoringDataCount >= minScoringRows) Some(sd)
      else {
        log.warn(s"Scoring dataset has $scoringDataCount rows, which is less than the minimum required of " +
          s"$minScoringRows. Only training data checks will be used.")
        None
      }
    }

    val scoringSummary = scoreData.map { sd =>
      val ss = computeFeatureStats(sd, rawFeatures, FeatureDistributionType.Scoring, Some(trainingSummary))
      log.info("Computed summary stats for scoring features")
      if (log.isDebugEnabled) {
        log.debug(ss.responseDistributions.mkString("\n"))
        log.debug(ss.predictorDistributions.mkString("\n"))
      }
      ss
    }

    val (rawFeatureFilterMetrics, exclusionReasons, featuresToDropNames, mapKeysToDrop) = getFeaturesToExclude(
      trainingSummary.predictorDistributions.filterNot(d => protectedFeatures.contains(d.name)),
      scoringSummary.toSeq.flatMap(_.predictorDistributions.filterNot(d => protectedFeatures.contains(d.name))),
      trainingSummary.correlationInfo)
    val (featuresToDrop, featuresToKeep) = rawFeatures.partition(rf => featuresToDropNames.contains(rf.name))
    val featuresToKeepNames = Array(DataFrameFieldNames.KeyFieldName) ++ featuresToKeep.map(_.name)

    require(featuresToKeep.count(!_.isResponse) > 0,
      "The raw feature filter has dropped all of your features, check your input data quality")

    val featuresDropped = trainData.drop(featuresToDropNames: _*)
    val mapsCleaned = featuresDropped.rdd.map { row =>
      val kept = featuresToKeepNames.map { fn =>
        mapKeysToDrop.get(fn) match {
          case Some(keysToDrop) => Option(row.getMapAny(fn)).map(_.filterKeys(k => !keysToDrop.contains(k))).orNull
          case None => row.getAny(fn)
        }
      }
      Row.fromSeq(kept)
    }

    val schema = StructType(featuresToKeepNames.map(featuresDropped.schema(_)))
    val cleanedData = spark.createDataFrame(mapsCleaned, schema).persist()
    trainData.unpersist()
    scoreData.map(_.unpersist())

    val rawFeatureFilterConfig = RawFeatureFilterConfig(
      minFill = minFill,
      maxFillDifference = maxFillDifference,
      maxFillRatioDiff = maxFillRatioDiff,
      maxJSDivergence = maxJSDivergence,
      maxCorrelation = maxCorrelation,
      correlationType = correlationType,
      jsDivergenceProtectedFeatures = jsDivergenceProtectedFeatures.toSeq,
      protectedFeatures = protectedFeatures.toSeq
    )

    val featureDistributions =
      trainingSummary.responseDistributions ++ trainingSummary.predictorDistributions ++
        scoringSummary.map(s => s.responseDistributions ++ s.predictorDistributions).getOrElse(Array.empty)

    val rawFeatureFilterResults = RawFeatureFilterResults(
      rawFeatureFilterConfig = rawFeatureFilterConfig,
      rawFeatureDistributions = featureDistributions,
      rawFeatureFilterMetrics = rawFeatureFilterMetrics,
      exclusionReasons = exclusionReasons
    )

    FilteredRawData(
      cleanedData = cleanedData,
      featuresToDrop = featuresToDrop,
      mapKeysToDrop = mapKeysToDrop,
      rawFeatureFilterResults = rawFeatureFilterResults
    )
  }
}


object RawFeatureFilter {

  /**
   * Default calculation for the hashing size for RFF (compare js distance) for text features
   *
   * @param summary summary info for feature (max, min, etc)
   * @param bins    number of bins to use
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

  // If there are not enough rows in the scoring set, we should not perform comparisons between the training and
  // scoring sets since they will not be reliable. Currently, this is set to the same as the minimum training size.
  val minScoringRowsDefault = 500
  val MaxCardinality = 500


  val stageName = classOf[RawFeatureFilter[_]].getSimpleName

  val uid = s"${stageName}_100000000000"
}

/**
 * Contains RFF filtered data, features to drop, and results from RFF
 *
 * @param cleanedData               RFF cleaned data
 * @param featuresToDrop            raw features dropped by RFF
 * @param mapKeysToDrop             keys in map features dropped by RFF
 * @param rawFeatureFilterResults   feature information calculated from the training data
 */
case class FilteredRawData
(
  cleanedData: DataFrame,
  featuresToDrop: Array[OPFeature],
  mapKeysToDrop: Map[String, Set[String]],
  rawFeatureFilterResults: RawFeatureFilterResults
) {

  /**
   * Feature distributions calculated from the training data
   */
  def trainingFeatureDistributions: Seq[FeatureDistribution] =
    rawFeatureFilterResults.rawFeatureDistributions.filter(_.`type` == FeatureDistributionType.Training)

  /**
   * Feature distributions calculated from the scoring data
   */
  def scoringFeatureDistributions: Seq[FeatureDistribution] =
    rawFeatureFilterResults.rawFeatureDistributions.filter(_.`type` == FeatureDistributionType.Scoring)

}
