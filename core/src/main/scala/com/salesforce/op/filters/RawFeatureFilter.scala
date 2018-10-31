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
import com.salesforce.op.filters.RawFeatureFilter._
import com.salesforce.op.filters.Summary._
import com.salesforce.op.readers.{DataFrameFieldNames, Reader}
import com.salesforce.op.stages.impl.feature.TimePeriod
import com.salesforce.op.stages.impl.preparators.CorrelationType
import com.salesforce.op.utils.spark.RichRow._
import com.salesforce.op.utils.stats.StreamingHistogram
import com.salesforce.op.utils.stats.StreamingHistogram.{StreamingHistogramBuilder => HistogramBuilder}
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.{Semigroup, Tuple2Semigroup}
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

  require(bins > 1 && bins <= FeatureDistribution.MaxBins, s"Invalid bin size $bins," +
    s" bins must be between 1 and ${FeatureDistribution.MaxBins}")
  require(minFill >= 0.0 && minFill <= 1.0, s"Invalid minFill size $minFill, minFill must be between 0 and 1")
  require(maxFillDifference >= 0.0 && maxFillDifference <= 1.0, s"Invalid maxFillDifference size $maxFillDifference," +
    s" maxFillDifference must be between 0 and 1")
  require(maxFillRatioDiff >= 0.0, s"Invalid maxFillRatioDiff size $maxFillRatioDiff," +
    s" maxFillRatioDiff must be greater than 0.0")
  require(maxJSDivergence >= 0.0 && maxJSDivergence <= 1.0, s"Invalid maxJSDivergence size $maxJSDivergence," +
    s" maxJSDivergence must be between 0 and 1")

  ClosureUtils.checkSerializable(textBinsFormula) match {
    case Failure(e) => throw new IllegalArgumentException("The argument textBinsFormula must be serializable", e)
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
    val preparedFeatures: RDD[PreparedFeatures] =
      data.rdd.map(PreparedFeatures(_, responses, predictors, timePeriod))
    val allFeatures: RDD[AllFeatures] = preparedFeatures.map(_.allFeatures)
    val textFeatures: RDD[Map[FeatureKey, Seq[String]]] = allFeatures.map(_._3)
    val (totalCount, responseSummaries, numericSummaries, textSummaries) =
      RawFeatureFilter.getAllSummaries(bins, allFeatures)
    // Have to use the training summaries do process scoring for comparison
    val responseDistributions: Map[FeatureKey, FeatureDistribution] =
      responseSummaries.map { case (key, sum) => key -> sum.getFeatureDistribution(key, totalCount) }
    val numericDistributions: Map[FeatureKey, FeatureDistribution] =
      numericSummaries.map { case (key, sum) => key -> sum.getFeatureDistribution(key, totalCount) }

    // This will initialize existing text summary hashing TFs
    textSummaries.foreach { case (_, textSum) => textSum.setHashingTF() }

    val textDistributions: Map[FeatureKey, FeatureDistribution] =
      RawFeatureFilter.getTextDistributions(textSummaries, textFeatures, totalCount)
    val allDistributions = AllDistributions(
      responseDistributions = responseDistributions,
      numericDistributions = numericDistributions,
      textDistributions = textDistributions)

    val correlationInfo: Map[FeatureKey, Map[FeatureKey, Double]] =
      allFeatureInfo.map(_.correlationInfo).getOrElse {
        val responseKeys: Array[FeatureKey] = responseDistributions.keySet.toArray
        val predictorKeys: Array[FeatureKey] = allDistributions.predictorDistributions.keySet.toArray
        val corrRDD: RDD[Vector] = preparedFeatures.map(_.getNullLabelLeakageVector(responseKeys, predictorKeys))
        val corrMatrix: Matrix = Statistics.corr(corrRDD, correlationType.sparkName)

        responseKeys.zipWithIndex.map { case (responseKey, i) =>
          responseKey -> predictorKeys.zipWithIndex.map { case (predictorKey, j) =>
            predictorKey -> min(abs(corrMatrix(i, j + responseKeys.length)), 1.0)
          }.toMap
        }.toMap
      }

    AllFeatureInformation(
      allDistributions = allDistributions,
      correlationInfo = correlationInfo)
  }

  /**
   * Take in the distribution summaries for datasets (scoring summary may be empty) and determine which
   * features should be dropped (including maps with all keys dropped) and which map keys need to be dropped
   * @param trainingDistribs summary of distributions for training data features
   * @param scoringDistribs summary of distributions for scoring data features (may be an empty seq)
   * @param correlationInfo info needed to determine feature to drop based on null label-leakage correlation
   * @return a list of feature names that should be dropped and a map of map keys that should be dropped
   *         Map(featureName -> key)
   */
  private[op] def getFeaturesToExclude(
    trainingDistribs: AllDistributions,
    scoringDistribs: Option[AllDistributions],
    correlationInfo: Map[FeatureKey, Map[FeatureKey, Double]]
  ): (Seq[String], Map[String, Set[String]]) = {
    val allNumericReasons = getAllReasons(
      trainingDistribs = trainingDistribs.numericDistributions,
      scoringDistribs = scoringDistribs.map(_.numericDistributions).getOrElse(Map()),
      correlationInfo = correlationInfo,
      minFill = minFill,
      maxCorrelation = maxCorrelation,
      maxFillDifference = maxFillDifference,
      maxFillRatioDiff = maxFillRatioDiff,
      maxJSDivergence = maxJSDivergence,
      jsDivergenceProtectedFeatures = jsDivergenceProtectedFeatures,
      jsDivergenceF = FeatureDistribution.densityJSDivergence)
    val allTextReasons = getAllReasons(
      trainingDistribs = trainingDistribs.textDistributions,
      scoringDistribs = scoringDistribs.map(_.textDistributions).getOrElse(Map()),
      correlationInfo = correlationInfo,
      minFill = minFill,
      maxCorrelation = maxCorrelation,
      maxFillDifference = maxFillDifference,
      maxFillRatioDiff = maxFillRatioDiff,
      maxJSDivergence = maxJSDivergence,
      jsDivergenceProtectedFeatures = jsDivergenceProtectedFeatures,
      jsDivergenceF = FeatureDistribution.massJSDivergence)

    val toDrop: Map[FeatureKey, (String, List[String])] = (allNumericReasons ++ allTextReasons).filter {
      case ((name, _), (featureDescription, reasons)) =>
        log.info(featureDescription)
        reasons.nonEmpty && !protectedFeatures(name)
    }
    val toDropKeySet: Set[FeatureKey] = toDrop.keySet

    toDrop.foreach { case (featureKey, (_, reasons)) =>
      log.info(s"Dropping feature $featureKey because:\n\t${reasons.mkString("\n\t")}\n")
    }

    val toDropFeatures: Map[String, Set[FeatureKey]] = toDropKeySet.groupBy(_._1)
    val toKeepFeatures: Map[String, Set[FeatureKey]] = trainingDistribs.predictorKeySet
      .union(scoringDistribs.map(_.predictorKeySet).getOrElse(Set()))
      .diff(toDropKeySet)
      .groupBy(_._1)

    val mapKeys = toKeepFeatures.keySet.intersect(toDropFeatures.keySet)
    val toDropNames = toDropFeatures.collect { case (k, _) if !mapKeys.contains(k) => k }.toSeq
    val toDropMapKeys = toDropFeatures.collect { case (k, v) if mapKeys.contains(k) => k -> v.flatMap(_._2).toSet }

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
    require(trainData.count() > 0, "RawFeatureFilter cannot work with empty training data")
    val trainingSummary = computeFeatureStats(trainData, rawFeatures)
    log.info("Computed summary stats for training features")
    if (log.isDebugEnabled) {
      log.debug(trainingSummary.allDistributions.responseDistributions.mkString("\n"))
      log.debug(trainingSummary.allDistributions.predictorDistributions.mkString("\n"))
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
        log.debug(ss.allDistributions.responseDistributions.mkString("\n"))
        log.debug(ss.allDistributions.predictorDistributions.mkString("\n"))
      }
      ss
    }

    val (featuresToDropNames, mapKeysToDrop) = getFeaturesToExclude(
      trainingSummary.allDistributions,
      scoringSummary.map(_.allDistributions),
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

    FilteredRawData(cleanedData, featuresToDrop, mapKeysToDrop, (
      trainingSummary.allDistributions.responseDistributions ++
        trainingSummary.allDistributions.predictorDistributions
    ).values.toArray)
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

  private def getAllSummaries(bins: Int, allFeatures: RDD[AllFeatures]): AllSummaries = {
    def apply(sum: AllSummaries, feat: AllFeatures): AllSummaries = {
      val (totalCount, responseSummaries, numericSummaries, textSummaries) = sum
      val (responseFeatures, numericFeatures, textFeatures) = feat

      def updateNumericSummaries(
        summaries: Map[FeatureKey, HistogramSummary],
        features: Map[FeatureKey, Seq[Double]]): Map[FeatureKey, HistogramSummary] =
        summaries.keySet.union(features.keySet).map { key =>
          val summary = summaries.get(key).getOrElse(new HistogramSummary(bins, bins * 10))
          val points = features.get(key).getOrElse(Seq())

          if (points.nonEmpty) key -> summary.update(points) else key -> summary
        }.toMap

      def updateTextSummaries(
        summaries: Map[FeatureKey, TextSummary],
        features: Map[FeatureKey, Seq[String]]): Map[FeatureKey, TextSummary] =
        summaries.keySet.union(features.keySet).map { key =>
          val summary = summaries.get(key).getOrElse(new TextSummary(_ => bins))

          features.get(key).map(text => key -> summary.update(text))
            .getOrElse(key -> summary)
        }.toMap

        val newResponseSummaries = updateNumericSummaries(responseSummaries, responseFeatures)
        val newNumericSummaries = updateNumericSummaries(numericSummaries, numericFeatures)
        val newTextSummaries = updateTextSummaries(textSummaries, textFeatures)

      (totalCount + 1.0, newResponseSummaries, newNumericSummaries, newTextSummaries)
    }

    def merge(sum1: AllSummaries, sum2: AllSummaries): AllSummaries = sum1 + sum2

    def empty: AllSummaries = (0.0, Map(), Map(), Map())

    allFeatures.aggregate(empty)(apply, merge)
  }

  def getTextDistributions(
    textSummaries: Map[FeatureKey, TextSummary],
    textFeatures: RDD[Map[FeatureKey, Seq[String]]],
    totalCount: Double): Map[FeatureKey, FeatureDistribution] = {
    def apply(sum: Map[FeatureKey, TextSummary], feat: Map[FeatureKey, Seq[String]]): Map[FeatureKey, TextSummary] =
      sum.map { case (key, s) => key -> s.updateDistribution(feat.get(key).getOrElse(Seq())) }

    def merge(
      sum1: Map[FeatureKey, TextSummary],
      sum2: Map[FeatureKey, TextSummary]): Map[FeatureKey, TextSummary] =
      sum1.keySet.union(sum2.keySet).map { key =>
        val updatedSum = (sum1.get(key), sum2.get(key)) match {
          case (Some(s1), Some(s2)) => s1.mergeDistribution(s2)
          case (Some(s1), _) => s1
          case (_, Some(s2)) => s2
          // This should never happen
          case _ => throw new RuntimeException(s"Unable to find text summary for feature key: $key")
        }

        key -> updatedSum
      }.toMap

      textFeatures.aggregate(textSummaries)(apply, merge)
        .map { case (key, sum) => key -> sum.getFeatureDistribution(key, totalCount) }
  }

  def getAllReasons(
    trainingDistribs: Map[FeatureKey, FeatureDistribution],
    scoringDistribs: Map[FeatureKey, FeatureDistribution],
    correlationInfo: Map[FeatureKey, Map[FeatureKey, Double]],
    minFill: Double,
    maxCorrelation: Double,
    maxFillDifference: Double,
    maxFillRatioDiff: Double,
    maxJSDivergence: Double,
    jsDivergenceProtectedFeatures: Set[String],
    jsDivergenceF: (FeatureDistribution, FeatureDistribution) => Double): Map[FeatureKey, (String, List[String])] = {

    val trainingReasons: Map[FeatureKey, List[String]] = for {
      trainingPair <- trainingDistribs
      (featureKey, trainingDistrib) = trainingPair
    } yield {
      val trainingUnfilled =
        if (trainingDistrib.fillRate < minFill) {
          Option(s"training fill rate did not meet min required ($minFill)")
        } else {
          None
        }
      val trainingNullLabelLeaker =
        if (correlationInfo.map(_._2.get(featureKey).exists(_ > maxCorrelation)).exists(identity(_))) {
          Option(s"null indicator correlation (absolute) exceeded max allowed ($maxCorrelation)")
        } else {
          None
        }

      featureKey -> List(trainingUnfilled, trainingNullLabelLeaker).flatten
    }
    val distribMismatches: Map[FeatureKey, (String, List[String])] = for {
      scoringPair <- scoringDistribs
      (featureKey, scoringDistrib) = scoringPair
      trainingDistrib <- trainingDistribs.get(featureKey)
    } yield {
      val trainingFillRate = trainingDistrib.fillRate
      val scoringFillRate = scoringDistrib.fillRate
      val jsDivergence = jsDivergenceF(trainingDistrib, scoringDistrib)
      val fillRatioDiff = trainingDistrib.relativeFillRatio(scoringDistrib)
      val fillRateDiff = trainingDistrib.relativeFillRate(scoringDistrib)

      val distribMetadata = s"\nTraining Data: $trainingDistrib\nScoring Data: $scoringDistrib\n" +
        s"Train Fill=$trainingFillRate, Score Fill=$scoringFillRate, JS Divergence=$jsDivergence, " +
        s"Fill Rate Difference=$fillRateDiff, Fill Ratio Difference=$fillRatioDiff\n"

      val scoringUnfilled =
        if (scoringFillRate < minFill) {
          Option(s"scoring fill rate did not meet min required ($minFill)")
        } else {
          None
        }
      val jsDivergenceCheck =
        if (!jsDivergenceProtectedFeatures.contains(featureKey._1) && jsDivergence > maxJSDivergence) {
          Option(s"JS Divergence exceeded max allowed ($maxJSDivergence)")
        } else {
          None
        }
      val fillRateCheck =
        if (fillRateDiff > maxFillDifference) {
          Option(s"fill rate difference exceeded max allowed ($maxFillDifference)")
        } else {
          None
        }
      val fillRatioCheck =
        if (fillRatioDiff > maxFillRatioDiff) {
          Option(s"fill ratio difference exceeded max allowed ($maxFillRatioDiff)")
        } else {
          None
        }

      featureKey ->
        (distribMetadata -> List(scoringUnfilled, jsDivergenceCheck, fillRateCheck, fillRatioCheck).flatten)
    }

    for {
      reasonsPair <- trainingReasons
      (featureKey, reasons) = reasonsPair
      descriptionPair <- distribMismatches.get(featureKey)
      (description, otherReasons) = descriptionPair
    } yield featureKey -> (description -> (reasons ++ otherReasons))
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
  featureDistributions: Array[FeatureDistribution]
)
