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

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.AllowLabelAsInput
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.stats.OpStatistics
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import enumeratum._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector => OldDenseVector, SparseVector => OldSparseVector, Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.slf4j.impl.Log4jLoggerAdapter

import scala.collection.mutable.ArrayBuffer
import scala.math.min
import scala.reflect.runtime.universe._


trait SanityCheckerParams extends DerivedFeatureFilterParams {

  final val sampleLowerLimit = new IntParam(
    parent = this, name = "sampleLowerLimit",
    doc = "Lower limit on number of samples in downsampled data set (note: sample limit will not be exact, due " +
      "to Spark's dataset sampling behavior)",
    isValid = ParamValidators.gtEq(lowerBound = 0)
  )
  def setSampleLowerLimit(value: Int): this.type = set(sampleLowerLimit, value)
  def getSampleLowerLimit: Int = $(sampleLowerLimit)

  final val sampleUpperLimit = new IntParam(
    parent = this, name = "sampleUpperLimit",
    doc = "Upper limit on number of samples in downsampled data set (note: sample limit will not be exact, due " +
      "to Spark's dataset sampling behavior)",
    isValid = ParamValidators.gtEq(lowerBound = 0)
  )
  def setSampleUpperLimit(value: Int): this.type = set(sampleUpperLimit, value)
  def getSampleUpperLimit: Int = $(sampleUpperLimit)

  final val checkSample = new DoubleParam(
    parent = this, name = "checkSample",
    doc = "Rate to downsample the data for statistical calculations (note: actual sampling will not be exact " +
      "due to Spark's dataset sampling behavior)",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = false, upperInclusive = true)
  )
  def setCheckSample(value: Double): this.type = set(checkSample, value)
  def getCheckSample: Double = $(checkSample)

  final val sampleSeed = new LongParam(
    parent = this, name = "sampleSeed",
    doc = "Seed to use when sampling"
  )
  def setSampleSeed(value: Long): this.type = set(sampleSeed, value)
  def getSampleSeed: Long = $(sampleSeed)

  final val maxCorrelation = new DoubleParam(
    parent = this, name = "maxCorrelation",
    doc = "Maximum correlation (absolute value) allowed between a feature in the feature vector and the label",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = true, upperInclusive = true)
  )
  def setMaxCorrelation(value: Double): this.type = set(maxCorrelation, value)
  def getMaxCorrelation: Double = $(maxCorrelation)

  final val minCorrelation = new DoubleParam(
    parent = this, name = "minCorrelation",
    doc = "Minimum correlation (absolute value) allowed between a feature in the feature vector and the label",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = true, upperInclusive = true)
  )
  def setMinCorrelation(value: Double): this.type = set(minCorrelation, value)
  def getMinCorrelation: Double = $(minCorrelation)

  final val correlationType = new Param[String](
    parent = this, name = "correlationType",
    doc = "Which coefficient to use for computing correlation"
  )
  def setCorrelationType(value: CorrelationType): this.type = set(correlationType, value.entryName)
  def getCorrelationType: CorrelationType = CorrelationType.withNameInsensitive($(correlationType))

  final val maxCramersV = new DoubleParam(
    parent = this, name = "maxCramersV",
    doc = "Maximum categorical correlation value (Cramer's V) allowed between a categorical feature in the" +
      "feature vector and a categorical label",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = true, upperInclusive = true)
  )
  def setMaxCramersV(value: Double): this.type = set(maxCramersV, value)
  def getMaxCramersV: Double = $(maxCramersV)

  final val categoricalLabel = new BooleanParam(
    parent = this, name = "categoricalLabel",
    doc = "If true, then label is treated as categorical (eg. Cramer's V will be calculated between it and " +
      "categorical features). If this is not set, then use a max class fraction of 0.1 to estimate whether label is" +
      "categorical or not."
  )
  def setCategoricalLabel(value: Boolean): this.type = set(categoricalLabel, value)
  def getCategoricalLabel: Boolean = $(categoricalLabel)

  final val removeFeatureGroup = new BooleanParam(
    parent = this, name = "removeFeatureGroup",
    doc = "If true, remove all features descended from a parent feature (parent is direct parent before" +
      " vectorization) which has derived features that meet the exclusion criteria."
  )
  def setRemoveFeatureGroup(value: Boolean): this.type = set(removeFeatureGroup, value)
  def getRemoveFeatureGroup: Boolean = $(removeFeatureGroup)

  final val protectTextSharedHash = new BooleanParam(
    parent = this, name = "protectTextSharedHash",
    doc = "If true, an individual hash is dropped/kept independently of related null indicators and" +
      " other hashes in the same shared hash space."
  )
  def setProtectTextSharedHash(value: Boolean): this.type = set(protectTextSharedHash, value)
  def getProtectTextSharedHash: Boolean = $(protectTextSharedHash)

  final val maxRuleConfidence = new DoubleParam(
    parent = this, name = "maxRuleConfidence",
    doc = "Maximum allowed confidence of association rules in categorical variables. A categorical variable will be " +
      "removed if there is a choice where the maximum confidence is above this threshold, and the support for that " +
      "choice is above the min rule support parameter, defined below.",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = true, upperInclusive = true)
  )
  def setMaxRuleConfidence(value: Double): this.type = set(maxRuleConfidence, value)
  def getMaxRuleConfidence: Double = $(maxRuleConfidence)

  final val minRequiredRuleSupport = new DoubleParam(
    parent = this, name = "minRuleSupport",
    doc = "Categoricals can be removed if an association rule is found between one of the choices and a categorical " +
      "label where the confidence of that rule is above maxRuleConfidence and the support fraction of that choice is " +
      "above minRuleSupport.",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = true, upperInclusive = true)
  )
  def setMinRequiredRuleSupport(value: Double): this.type = set(minRequiredRuleSupport, value)
  def getMinRequiredRuleSupport: Double = $(minRequiredRuleSupport)

  final val featureLabelCorrOnly = new BooleanParam(
    parent = this, name = "featureLabelCorrOnly",
    doc = "If true, then only calculate the correlations between the features and the label. Otherwise, calculate " +
      "the entire correlation matrix, which includes all feature-feature correlations."
  )
  def setFeatureLabelCorrOnly(value: Boolean): this.type = set(featureLabelCorrOnly, value)
  def getFeatureLabelCorrOnly: Boolean = $(featureLabelCorrOnly)

  final val correlationExclusion: Param[String] = new Param[String](this, "correlationExclusion",
    "Setting for what categories of feature vector columns to exclude from the correlation calculation",
    (value: String) => CorrelationExclusion.withNameInsensitiveOption(value).isDefined
  )
  def setCorrelationExclusion(v: CorrelationExclusion): this.type = set(correlationExclusion, v.entryName)
  def getCorrelationExclusion: CorrelationExclusion = CorrelationExclusion.withNameInsensitive($(correlationExclusion))

  setDefault(
    checkSample -> SanityChecker.CheckSample,
    sampleSeed -> SanityChecker.SampleSeed,
    sampleLowerLimit -> SanityChecker.SampleLowerLimit,
    sampleUpperLimit -> SanityChecker.SampleUpperLimit,
    maxCorrelation -> SanityChecker.MaxCorrelation,
    minCorrelation -> SanityChecker.MinCorrelation,
    minVariance -> SanityChecker.MinVariance,
    maxCramersV -> SanityChecker.MaxCramersV,
    removeBadFeatures -> SanityChecker.RemoveBadFeatures,
    removeFeatureGroup -> SanityChecker.RemoveFeatureGroup,
    protectTextSharedHash -> SanityChecker.ProtectTextSharedHash,
    correlationType -> SanityChecker.CorrelationTypeDefault.entryName,
    maxRuleConfidence -> SanityChecker.MaxRuleConfidence,
    minRequiredRuleSupport -> SanityChecker.MinRequiredRuleSupport,
    featureLabelCorrOnly -> SanityChecker.FeatureLabelCorrOnly,
    correlationExclusion -> SanityChecker.CorrelationExclusionDefault.entryName
  )
}


/**
 * The SanityChecker checks for potential problems with computed features in a supervised learning setting.
 *
 * There is an Estimator step, which outputs statistics on the incoming data, as well as the names of features
 * which should be dropped from the feature vector.  The transformer step applies the action of actually
 * removing the offending features from the feature vector.
 */
class SanityChecker(uid: String = UID[SanityChecker])
  extends BinaryEstimator[RealNN, OPVector, OPVector](
    operationName = classOf[SanityChecker].getSimpleName, uid = uid
  ) with SanityCheckerParams with AllowLabelAsInput[OPVector] {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  /**
   * Calculates an Array of CategoricalGroupStats objects, each one corresponding to a single categorical feature.
   *
   * @param sampleSize    Number of data points
   * @param featureSize   Length of the feature vector
   * @param columnMeta    Sequence of OpVectorColumnMetadata, mainly for determining which feature vector indices
   *                      correspond to categorical features
   * @param data          RDD of data in the form of (label, featureVector)
   * @return              Array of CategoricalGroupStats objects, one per categorical feature
   */
  private def categoricalTests(
    sampleSize: Long,
    featureSize: Int,
    columnMeta: Seq[OpVectorColumnMetadata],
    data: RDD[(Double, OPVector#Value)]
  ): Array[CategoricalGroupStats] = {
    // Figure out which columns correspond to MultiPickList values so that we can make the "OTHER" columns at most 1 so
    // that we can still use contingency matrices to calculate Cramer's V values
    val multiPickListIndices = columnMeta.zipWithIndex.collect {
      case (col, index) if col.hasParentOfSubType[MultiPickList] => index
    }.toSet

    // Group by label and then add in a 1.0 so we can get the total occurrences for each label in one reduction
    val contingencyData = data.map { case (label, featuresVector) =>
      val contingency = new Array[Double](featuresVector.size + 1)
      featuresVector.foreachActive { case (i, value) =>
        contingency(i) = if (multiPickListIndices.contains(i)) math.min(value, 1.0) else value
      }
      contingency(featuresVector.size) = 1.0
      label -> contingency
    }.reduceByKey(_ + _).persist()

    // Only calculate this if the label is categorical. Either the user specifies the label is categorical with
    // the categoricalLabel param, or if that is not set we assume the label is categorical if the number
    // of distinct labels is less than the min of 100 and sample size * 0.1
    val distinctLabels = contingencyData.count()
    val stats =
      if (isDefined(categoricalLabel) && $(categoricalLabel) || distinctLabels < min(100.0, sampleSize * 0.1)) {
        val contingencyWithKeys = contingencyData.collect()
        val contingency = contingencyWithKeys.sortBy(_._1).map { case (_, vector) => vector }

        logInfo("Label is assumed to be categorical since either categoricalLabel = true or " +
          "number of distinct labels < count * 0.1")

        // Only perform Cramer's V calculation on columns that have an grouping and indicatorValue defined (right
        // now, the only things that will have indicatorValue defined and grouping be None is numeric maps)
        val columnsWithIndicator = columnMeta.filter(f => f.grouping.isDefined && f.indicatorValue.isDefined)
        val colIndicesByGrouping =
          columnsWithIndicator
            .map { meta => meta.featureGroup().get -> meta }
            .groupBy(_._1)
            // Keep track of the group, column name, column index, and whether the parent was a MultiPickList or not
            .map{ case (group, cols) =>
              val repeats = cols.map(c => (c._2.indicatorValue, c._2.index)).groupBy(_._1)
                .collect{ case (_, seq) if seq.length > 1 => seq.tail.map(_._2) } // only first used in stats
                .flatten.toSet
              val colsCleaned = cols.map(_._2).filterNot(c => repeats.contains(c.index))
              (group, colsCleaned.map(_.makeColName()), colsCleaned.map(_.index),
                colsCleaned.exists(_.hasParentOfSubType[MultiPickList]))
          }

        colIndicesByGrouping.map {
          case (group, colNames, valueIndices, isMpl) =>
            val groupContingency =
              if (valueIndices.length == 1) {
                // parentFeatureNames only has a single indicator column, construct the other from label sums
                contingency.flatMap(features => {
                  val indicatorSum = valueIndices.map(features.apply)
                  indicatorSum ++ indicatorSum.map(features.last - _)
                })
              } else {
                contingency.flatMap { features => valueIndices.map(features.apply) }
              }

            // columns are label value, rows are feature value
            val contingencyMatrix =
              if (valueIndices.length == 1) {
                new DenseMatrix(2, groupContingency.length / 2, groupContingency)
              } else {
                new DenseMatrix(valueIndices.length, groupContingency.length / valueIndices.length, groupContingency)
              }

            val cStats =
              if (isMpl) {
                val labelCounts = contingency.map(_.last)
                OpStatistics.contingencyStatsFromMultiPickList(contingencyMatrix, labelCounts)
              } else OpStatistics.contingencyStats(contingencyMatrix)

            CategoricalGroupStats(
              group = group,
              categoricalFeatures = colNames.toArray,
              contingencyMatrix = cStats.contingencyMatrix,
              pointwiseMutualInfo = cStats.pointwiseMutualInfo,
              cramersV = cStats.chiSquaredResults.cramersV,
              mutualInfo = cStats.mutualInfo,
              maxRuleConfidences = cStats.confidenceResults.maxConfidences,
              supports = cStats.confidenceResults.supports
            )
        }.toArray
      } else {
        logInfo(s"Label is assumed to be continuous since number of distinct labels = $distinctLabels" +
          s"which is greater than 10% the size of the sample $sampleSize skipping calculation of Cramer's V")
        Array.empty[CategoricalGroupStats]
      }
    contingencyData.unpersist(blocking = false)
    stats
  }

  /**
   * Estimate of a fraction of data that is being checked.
   *
   * @param totalSize
   * @return
   */
  def fraction(totalSize: Long): Double = {
    val ckSample = $(checkSample)
    val minFraction = math.min(1.0, $(sampleLowerLimit).toDouble / totalSize)
    val maxFraction = math.max(0.0, $(sampleUpperLimit).toDouble / totalSize)
    math.max(math.min(ckSample, maxFraction), minFraction)
  }

  /**
   * The SanityChecker's core 'estimator' function, computes statistics on the features and the
   * list of features to be removed.
   */
  override def fitFn(data: Dataset[(RealNN#Value, OPVector#Value)]): BinaryModel[RealNN, OPVector, OPVector] = {
    // Set the desired log level
    if (isSet(logLevel)) {
      Option(log).collect { case l: Log4jLoggerAdapter =>
        LogManager.getLogger(l.getName).setLevel(Level.toLevel($(logLevel)))
      }
    }
    val sampSeed = $(sampleSeed)
    val removeBad = $(removeBadFeatures)
    val corrType = $(correlationType)

    val dataCount = data.count()
    val sampleFraction = fraction(dataCount)
    val sampleData: RDD[(Double, OPVector#Value)] = {
      if (sampleFraction < 1.0) {
        logDebug(s"Sampling the data for Sanity Checker with sample $sampleFraction and seed $sampSeed")
        data.sample(withReplacement = false, fraction = sampleFraction, seed = sampSeed).rdd
      } else {
        logDebug(s"NOT sampling the data for Sanity Checker, since the calculated check sample is $sampleFraction")
        data.rdd
      }
    } map {
      case (Some(label), features) => label -> features
      case _ =>
        // Should not happen, since label is of RealNN feature type
        throw new IllegalArgumentException("Sanity checker input missing label for row")
    }
    sampleData.persist()

    logDebug("Getting vector rows")
    val vectorRows: RDD[OldVector] = sampleData.map {
      case (0.0, sparse: SparseVector) =>
        OldVectors.sparse(sparse.size + 1, sparse.indices, sparse.values)
      case (label, sparse: SparseVector) =>
        OldVectors.sparse(sparse.size + 1, sparse.indices :+ sparse.size, sparse.values :+ label)
      case (label, dense: DenseVector) =>
        OldVectors.dense(dense.toArray :+ label)
    }.persist()

    logDebug("Calculating columns stats")
    val colStats = Statistics.colStats(vectorRows)
    val count = colStats.count
    require(count > 0, "Sample size cannot be zero")

    val featureSize = vectorRows.first().size - 1
    require(featureSize > 0, "Feature vector passed in is empty, check your vectorizers")

    // handle any possible serialization errors if users give us wrong metadata
    val vectorMeta = try {
      OpVectorMetadata(getInputSchema()(in2.name))
    } catch {
      case e: NoSuchElementException =>
        throw new IllegalArgumentException("Vector input metadata is malformed: ", e)
    }

    require(featureSize == vectorMeta.size,
      s"Number of columns in vector metadata (${vectorMeta.size}) did not match number of columns in data" +
        s"($featureSize), check your vectorizers \n metadata=$vectorMeta")
    val vectorMetaColumns = vectorMeta.columns
    val featureNames = vectorMetaColumns.map(_.makeColName())

    val (corrIndices, vectorRowsForCorr) = if ($(correlationExclusion) == CorrelationExclusion.HashedText.entryName) {
      val hashedIndices = vectorMetaColumns
        .filter(f =>
          // Indices are determined to be hashed if they come from Text/TextArea types (or their maps) and don't have
          // an indicator group or indicator value (indicating that they are not pivoted out by the SmartTextVectorizer
          // TODO: Find a better way to do this with the feature history
          f.grouping.isEmpty && f.indicatorValue.isEmpty &&
            f.parentFeatureType.exists { t =>
              val tt = FeatureType.featureTypeTag(t)
              tt.tpe =:= typeTag[Text].tpe || tt.tpe =:= typeTag[TextArea].tpe ||
                tt.tpe =:= typeTag[TextMap].tpe || tt.tpe =:= typeTag[TextAreaMap].tpe
            }
        )
        .map(f => f.index)
      val localCorrIndices = (0 until featureSize + 1).diff(hashedIndices).toArray

      logDebug(s"Ignoring correlations for hashed text features - out of $featureSize feature vector elements, using " +
        s"${localCorrIndices.length} elements in the correlation matrix calculation")

      // Exclude feature vector entries coming from hashed text features if requested
      val localVectorRowsForCorr = vectorRows.map {
        case v: OldDenseVector =>
          val res = localCorrIndices.map(v.apply)
          OldVectors.dense(res)
        case v: OldSparseVector => {
          val res = new ArrayBuffer[(Int, Double)]()
          v.foreachActive((i, v) => if (localCorrIndices.contains(i)) res += localCorrIndices.indexOf(i) -> v)
          OldVectors.sparse(localCorrIndices.length, res).compressed
        }
      }
      (localCorrIndices, localVectorRowsForCorr)
    }
    // Make sure to include the label, correlation matrix has dimensions (featureSize + 1, featureSize + 1)
    else ((0 until featureSize + 1).toArray, vectorRows)
    val numCorrIndices = corrIndices.length

    // TODO: We are still calculating the full correlation matrix when featureLabelCorrOnly is false, but are not
    // TODO: storing it anywhere. If we want access to the inter-feature correlations then need to refactor this a bit.
    val corrsWithLabel = if ($(featureLabelCorrOnly)) {
      OpStatistics.computeCorrelationsWithLabel(vectorRowsForCorr, colStats, count)
    }
    else Statistics.corr(vectorRowsForCorr, getCorrelationType.sparkName).rowIter
      .map(_.apply(numCorrIndices - 1)).toArray

    // Only calculate this if the label is categorical, so ignore if user has flagged label as not categorical
    val categoricalStats =
      if (isDefined(categoricalLabel) && !$(categoricalLabel)) {
        Array.empty[CategoricalGroupStats]
      } else {
        logDebug("Attempting to calculate Cramer's V between each categorical feature and the label")
        categoricalTests(count, featureSize, vectorMetaColumns, sampleData)
      }

    logDebug("Logging all statistics")
    val stats = DerivedFeatureFilterUtils.makeColumnStatistics(
      vectorMetaColumns,
      colStats,
      Option((in1.name, featureSize)), // label column goes at end of vector
      corrsWithLabel,
      corrIndices,
      categoricalStats
    )
    stats.foreach { stat => logDebug(stat.toString) }

    logDebug("Calculating features to remove")
    val (toDropFeatures, warnings) = if (removeBad) {
      DerivedFeatureFilterUtils.getFeaturesToDrop(
        stats,
        $(minVariance),
        $(minCorrelation),
        $(maxCorrelation),
        $(maxCramersV),
        $(maxRuleConfidence),
        $(minRequiredRuleSupport),
        $(removeFeatureGroup),
        $(protectTextSharedHash)
      ).unzip
    } else (Array.empty[ColumnStatistics], Array.empty[String])
    warnings.foreach { warning => logWarning(warning) }

    val toDropSet = toDropFeatures.flatMap(_.column).toSet
    val outputFeatures = vectorMetaColumns.filterNot { col => toDropSet.contains(col) }
    val indicesToKeep = outputFeatures.map(_.index)

    val outputMeta = OpVectorMetadata(getOutputFeatureName, outputFeatures, vectorMeta.history)

    // TODO: Refactor so that everything is constructed directly from our Array[ColumnStatistics], stats
    val summary = new SanityCheckerSummary(
      stats = stats,
      catStats = categoricalStats,
      dropped = toDropFeatures.map(_.name),
      colStats = colStats,
      names = featureNames :+ in1.name,
      correlationType = CorrelationType.withNameInsensitive(corrType),
      sample = sampleFraction
    )
    setMetadata(outputMeta.toMetadata.withSummaryMetadata(summary.toMetadata()))

    sampleData.unpersist(blocking = false)
    vectorRows.unpersist(blocking = false)

    require(indicesToKeep.length > 0,
      "The sanity checker has dropped all of your features, check your input data quality")

    new SanityCheckerModel(
      indicesToKeep = indicesToKeep,
      removeBadFeatures = removeBad,
      operationName = operationName,
      uid = uid
    )
  }
}

final class SanityCheckerModel private[op]
(
  val indicesToKeep: Array[Int],
  val removeBadFeatures: Boolean,
  operationName: String,
  uid: String
) extends BinaryModel[RealNN, OPVector, OPVector](operationName = operationName, uid = uid)
  with AllowLabelAsInput[OPVector] {

  /**
   * The SanityChecker's core 'transformer' function, which removes the features recommended to be removed.
   */
  def transformFn: (RealNN, OPVector) => OPVector = (label, feature) => {
    DerivedFeatureFilterUtils.removeFeatures(indicesToKeep, removeBadFeatures)(feature)
  }
}

object SanityChecker {
  val CheckSample = 1.0
  val SampleLowerLimit = 1E3.toInt
  val SampleUpperLimit = 1E6.toInt
  val MaxCorrelation = 0.95
  val MinCorrelation = 0.0
  val MinVariance = 1E-5
  val MaxCramersV = 0.95
  val RemoveBadFeatures = false
  val RemoveFeatureGroup = true
  val ProtectTextSharedHash = false
  val CorrelationTypeDefault = CorrelationType.Pearson
  // These settings will make the maxRuleConfidence check off by default
  val MaxRuleConfidence = 1.0
  val MinRequiredRuleSupport = 1.0
  val FeatureLabelCorrOnly = false
  val CorrelationExclusionDefault = CorrelationExclusion.NoExclusion

  def SampleSeed: Long = util.Random.nextLong() // scalastyle:off method.name
}

/**
 * Represents a kind of correlation coefficient.
 *
 * @param sparkName The spark name of the correlation coefficient
 */
sealed abstract class CorrelationType(val sparkName: String) extends EnumEntry with Serializable

object CorrelationType extends Enum[CorrelationType] {
  val values: Seq[CorrelationType] = findValues
  /**
   * Compute with Pearson correlation
   *
   * @see https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
   */
  case object Pearson extends CorrelationType("pearson")
  /**
   * Compute with Spearman's rank-order correlation
   *
   * @see https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
   */
  case object Spearman extends CorrelationType("spearman")

  /**
   * Compute with Spearman's rank-order correlation
   *
   * @see https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
   */
  case class Custom(name: String, spark: String) extends CorrelationType(spark)
}

/**
 * Categories of feature vector columns to exclude from the feature-label correlation matrix (or just array of
 * feature-label correlations) calculated inSanityChecker.
 */
sealed trait CorrelationExclusion extends EnumEntry with Serializable

object CorrelationExclusion extends Enum[CorrelationExclusion] {
  val values: Seq[CorrelationExclusion] = findValues

  /**
   * Don't exclude any feature vector columns from the correlation calculation
   */
  case object NoExclusion extends CorrelationExclusion

  /**
   * Exclude columns coming from hashed text features
   */
  case object HashedText extends CorrelationExclusion
}
