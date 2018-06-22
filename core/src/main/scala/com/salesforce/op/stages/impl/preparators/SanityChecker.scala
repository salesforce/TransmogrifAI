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

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.AllowLabelAsInput
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.impl.preparators.CorrelationType.Pearson
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.stats.OpStatistics
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import enumeratum._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors => NewVectors}
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.slf4j.impl.Log4jLoggerAdapter

import scala.math.min
import scala.util.Try


trait SanityCheckerParams extends Params {

  final val logLevel = new Param[String](
    parent = this, name = "logLevel",
    doc = "sets log level (INFO, WARN, ERROR, DEBUG etc.)",
    isValid = (s: String) => Try(Level.toLevel(s)).isSuccess
  )
  private[op] def setLogLevel(level: Level): this.type = set(logLevel, level.toString)

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

  final val removeBadFeatures = new BooleanParam(
    parent = this, name = "removeBadFeatures",
    doc = "If set to true, this will automatically remove all the bad features from the feature vector"
  )
  def setRemoveBadFeatures(value: Boolean): this.type = set(removeBadFeatures, value)
  def getRemoveBadFeatures: Boolean = $(removeBadFeatures)

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

  final val correlationType = new Param[CorrelationType](
    parent = this, name = "correlationType",
    doc = "Which coefficient to use for computing correlation"
  )
  def setCorrelationType(value: CorrelationType): this.type = set(correlationType, value)
  def getCorrelationType: CorrelationType = $(correlationType)

  final val minVariance = new DoubleParam(
    parent = this, name = "minVariance",
    doc = "Minimum amount of variance allowed for each feature and label"
  )
  def setMinVariance(value: Double): this.type = set(minVariance, value)
  def getMinVariance: Double = $(minVariance)

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
    correlationType -> SanityChecker.CorrelationType,
    maxRuleConfidence -> SanityChecker.MaxRuleConfidence,
    minRequiredRuleSupport -> SanityChecker.MinRequiredRuleSupport
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

  private def makeColumnStatistics(
    metaCols: Seq[OpVectorColumnMetadata],
    labelColumnIndex: Int,
    corrMatrix: Matrix,
    statsSummary: MultivariateStatisticalSummary,
    categoricalStats: Array[CategoricalGroupStats]
  ): Array[ColumnStatistics] = {
    // precompute all statistics to avoid rebuilding the vectors every time
    val means = statsSummary.mean
    val maxs = statsSummary.max
    val mins = statsSummary.min
    val count = statsSummary.count
    val variances = statsSummary.variance
    val cramersVMap = categoricalStats.flatMap(f => f.categoricalFeatures.map(c => c -> f.cramersV))
      .toMap[String, Double]

    // build map from indicator group name to any null indicator it may have
    val nullGroups = for {
      col <- metaCols
      if col.isNullIndicator
        indicatorGroup <- col.indicatorGroup
      } yield (indicatorGroup, (col, col.index))

    nullGroups.groupBy(_._1).foreach {
      case (group, cols) =>
        require(cols.length == 1, s"Vector column $group has multiple null indicator fields: $cols")
    }

    def maxByParent(seq: Seq[(String, Double)]) = seq.groupBy(_._1).map{ case(k, v) =>
      // Filter out the NaNs because max(3.4, NaN) = NaN, and we still want the keep the largest correlation
      k -> v.filterNot(_._2.isNaN).foldLeft(0.0)((a, b) => math.max(a, math.abs(b._2)))
    }

    def corrParentMap(fn: OpVectorColumnMetadata => Seq[String]) =
      maxByParent(metaCols.flatMap(c => fn(c).map(_ -> corrMatrix(c.index, labelColumnIndex))))

    def cramersVParentMap(fn: OpVectorColumnMetadata => Seq[String]) = {
      val parentMap = metaCols.flatMap{ c => fn(c).map(c.makeColName() -> _) }.toMap
      maxByParent(cramersVMap.toSeq.map { case (k, v) => parentMap(k) -> v })
    }

    val corrParent = corrParentMap(_.parentNamesWithMapKeys())
    val corrParentNoKeys = corrParentMap(_.parentFeatureName)

    val cramersVParent = cramersVParentMap(_.parentNamesWithMapKeys())
    val cramersVParentNoKeys = cramersVParentMap(_.parentFeatureName)

    // These are the categorical features that are alone in their feature group. This means that they are
    // null indicator columns coming from non-categorical features, so they correspond to a 2x2 contingency matrix
    // and thus two support and maxRuleConfidence values
    val supportMap = categoricalStats.flatMap(f =>
      if (f.categoricalFeatures.length == 1) Array(f.categoricalFeatures.head -> f.supports.toSeq)
      else {
        f.categoricalFeatures.zip(f.supports).map(f => f._1 -> Seq(f._2))
      }).toMap
    val maxRuleConfMap = categoricalStats.flatMap(f =>
      if (f.categoricalFeatures.length == 1) Array(f.categoricalFeatures.head -> f.maxRuleConfidences.toSeq)
      else {
        f.categoricalFeatures.zip(f.maxRuleConfidences).map(f => f._1 -> Seq(f._2))
      }).toMap

    // TODO: For Hashing vectorizers, there is no indicator group, so check for parentNamesWithMapKeys()
    // inside cramersVParent first and then check without keys for removal. This will over-remove features (eg.
    // an entire map), but should only affect custom map vectorizers that don't set indicator groups on columns.
    def getParentValue(col: OpVectorColumnMetadata, check1: Map[String, Double], check2: Map[String, Double]) =
    col.parentNamesWithMapKeys().flatMap( k => check1.get(k).orElse(check2.get(k)) ).reduceOption(_ max _)

    val featuresStats = metaCols.map {
      col =>
        val i = col.index
        val name = col.makeColName()
        ColumnStatistics(
          name = name,
          column = Some(col),
          isLabel = false,
          count = count,
          mean = means(i),
          min = mins(i),
          max = maxs(i),
          variance = variances(i),
          corrLabel = Option(corrMatrix(i, labelColumnIndex)),
          cramersV = cramersVMap.get(name),
          parentCorr = getParentValue(col, corrParent, corrParentNoKeys),
          parentCramersV = getParentValue(col, cramersVParent, cramersVParentNoKeys),
          maxRuleConfidences = maxRuleConfMap.getOrElse(name, Seq.empty),
          supports = supportMap.getOrElse(name, Seq.empty)
        )
    }
    val labelStats = ColumnStatistics(
      name = in1.name,
      column = None,
      isLabel = true,
      count = count,
      mean = means(labelColumnIndex),
      min = mins(labelColumnIndex),
      max = maxs(labelColumnIndex),
      variance = variances(labelColumnIndex),
      corrLabel = None,
      cramersV = None,
      parentCorr = None,
      parentCramersV = None,
      maxRuleConfidences = Seq.empty,
      supports = Seq.empty
    )
    (labelStats +: featuresStats).toArray
  }

  private def getFeaturesToDrop(stats: Array[ColumnStatistics]): Array[ColumnStatistics] = {
    val minVar = $(minVariance)
    val minCorr = $(minCorrelation)
    val maxCorr = $(maxCorrelation)
    val maxCramV = $(maxCramersV)
    val maxRuleConf = $(maxRuleConfidence)
    val minReqRuleSupport = $(minRequiredRuleSupport)
    val removeFromParent = $(removeFeatureGroup)
    val textSharedHashProtected = $(protectTextSharedHash)

    // Calculate groups to remove separately. This is for more complicated checks where you can't determine whether
    // to remove a feature from a single column stats (eg. associate rule confidence/support check)
    val groupByGroups = stats.groupBy(_.column.flatMap(_.indicatorGroup))
    val ruleConfGroupsToDrop = groupByGroups.toSeq.flatMap{
      case (Some(group), colStats) =>
        val colsToRemove = colStats.filter(f =>
          f.maxRuleConfidences.zip(f.supports).exists{
            case (maxConf, sup) => (maxConf > maxRuleConf) && (sup > minReqRuleSupport)
          })
        if (colsToRemove.nonEmpty) Option(group) else None

      case _ => None
    }

    for {
      col <- stats
      reasons = col.reasonsToRemove(
        minVariance = minVar,
        minCorrelation = minCorr,
        maxCorrelation = maxCorr,
        maxCramersV = maxCramV,
        maxRuleConfidence = maxRuleConf,
        minRequiredRuleSupport = minReqRuleSupport,
        removeFeatureGroup = removeFromParent,
        protectTextSharedHash = textSharedHashProtected,
        removedGroups = ruleConfGroupsToDrop
      )
      if reasons.nonEmpty
    } yield {
      logWarning(s"Removing ${col.name} due to: ${reasons.mkString(",")}")
      col
    }
  }

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

      // Only perform Cramer's V calculation on columns that have an indicatorGroup and indicatorValue defined (right
      // now, the only things that will have indicatorValue defined and indicatorGroup be None is numeric maps)
      val columnsWithIndicator = columnMeta.filter(f => f.indicatorGroup.isDefined && f.indicatorValue.isDefined)
      val colIndicesByIndicatorGroup =
        columnsWithIndicator
          .map { meta => meta.indicatorGroup.get -> meta }
          .groupBy(_._1)
          // Keep track of the group, column name, column index, and whether the parent was a MultiPickList or not
          .map { case (group, cols) => (group, cols.map(_._2.makeColName()), cols.map(_._2.index),
            cols.exists(_._2.hasParentOfSubType[MultiPickList]))
          }

      colIndicesByIndicatorGroup.map {
        case (group, colNames, valueIndices, isMpl) =>
          val groupContingency =
            if (valueIndices.length == 1) {
              // parentFeatureNames only has a single indicator column, construct the other from label sums
              contingency.flatMap(features => {
                val indicatorSum = valueIndices.map(features.apply)
                indicatorSum ++ indicatorSum.map(features.last - _)
              })
            } else contingency.flatMap { features => valueIndices.map(features.apply) }

          // columns are label value, rows are feature value
          val contingencyMatrix = if (valueIndices.length == 1) {
            new DenseMatrix(2, groupContingency.length / 2, groupContingency)
          }
          else new DenseMatrix(valueIndices.length, groupContingency.length / valueIndices.length, groupContingency)

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
        logInfo(s"Sampling the data for Sanity Checker with sample $sampleFraction and seed $sampSeed")
        data.sample(withReplacement = false, fraction = sampleFraction, seed = sampSeed).rdd
      } else {
        logInfo(s"NOT sampling the data for Sanity Checker, since the calculated check sample is $sampleFraction")
        data.rdd
      }
    } map {
      case (Some(label), features) => label -> features
      case _ =>
        // Should not happen, since label is of RealNN feature type
        throw new IllegalArgumentException("Sanity checker input missing label for row")
    }
    sampleData.persist()

    logInfo("Getting vector rows")
    val vectorRows: RDD[OldVector] = sampleData.map {
      case (0.0, sparse: SparseVector) =>
        OldVectors.sparse(sparse.size + 1, sparse.indices, sparse.values)
      case (label, sparse: SparseVector) =>
        OldVectors.sparse(sparse.size + 1, sparse.indices :+ sparse.size, sparse.values :+ label)
      case (label, dense: DenseVector) =>
        OldVectors.dense(dense.toArray :+ label)
    }.persist()

    logInfo("Calculating columns stats")
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
      "Number of columns in vector metadata did not match number of columns in data, check your vectorizers")
    val vectorMetaColumns = vectorMeta.columns
    val featureNames = vectorMetaColumns.map(_.makeColName())

    logInfo(s"Calculating ${corrType.sparkName} correlations")
    val covariance = Statistics.corr(vectorRows, corrType.sparkName)

    // Only calculate this if the label is categorical, so ignore if user has flagged label as not categorical
    val categoricalStats =
      if (isDefined(categoricalLabel) && !$(categoricalLabel)) {
        Array.empty[CategoricalGroupStats]
      } else {
        logInfo("Attempting to calculate Cramer's V between each categorical feature and the label")
        categoricalTests(count, featureSize, vectorMetaColumns, sampleData)
      }

    logInfo("Logging all statistics")
    val stats = makeColumnStatistics(
      vectorMetaColumns,
      labelColumnIndex = featureSize, // label column goes at end of vector
      covariance, colStats, categoricalStats
    )
    stats.foreach { stat => logInfo(stat.toString) }

    logInfo("Calculating features to remove")
    val toDropFeatures = if (removeBad) getFeaturesToDrop(stats) else Array.empty[ColumnStatistics]
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
      correlationType = corrType,
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
    if (!removeBadFeatures) feature
    else {
      val vals = new Array[Double](indicesToKeep.length)
      feature.value.foreachActive((i, v) => {
        val k = indicesToKeep.indexOf(i)
        if (k >= 0) vals(k) = v
      })
      NewVectors.dense(vals).compressed.toOPVector
    }
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
  val CorrelationType = Pearson
  // These settings will make the maxRuleConfidence check off by default
  val MaxRuleConfidence = 1.0
  val MinRequiredRuleSupport = 1.0

  def SampleSeed: Long = util.Random.nextLong() // scalastyle:off method.name
}

/**
 * Holds information related to the statistics of a column in the feature vector.
 *
 * [[column]] will always be present if not a label, and will not be present if this is a label.
 */
private[op] case class ColumnStatistics
(
  name: String,
  column: Option[OpVectorColumnMetadata],
  isLabel: Boolean,
  count: Long,
  mean: Double,
  min: Double,
  max: Double,
  variance: Double,
  corrLabel: Option[Double],
  cramersV: Option[Double],
  parentCorr: Option[Double],
  parentCramersV: Option[Double],
  // Need to be able to hold up to two maxRuleConfidences or supports for the case of nullIndicator columns coming
  // from non-categorical features (since they will correspond to a 2x2 contingency matrix)
  maxRuleConfidences: Seq[Double],
  supports: Seq[Double]
) {

  /**
   * Given a minimum variance, maximum variance, and maximum correlation, decide if there is a reason to remove
   * this column. If so, return a list of the reasons why. If not, then return an empty list.
   *
   * @param minVariance         Minimum variance
   * @param maxCorrelation      Maximum correlation
   * @param minCorrelation      Minimum correlation
   * @param maxCramersV         Maximum Cramer's V value
   * @param maxRuleConfidence   Minimum association rule confidence between
   * @param minRequiredRuleSupport  Minimum required support to throw away a group
   * @param removeFeatureGroup   Whether to remove entire feature group when any group value is flagged for removal
   * @param protectTextSharedHash   Whether to protect text shared hash from related null indicator and other hashes
   * @param removedGroups       Pre-determined feature groups to remove (eg. via maxRuleConfidence)
   * @return List[String] if reason to remove, nil otherwise
   */
  def reasonsToRemove(
    minVariance: Double,
    maxCorrelation: Double,
    minCorrelation: Double,
    maxCramersV: Double,
    maxRuleConfidence: Double,
    minRequiredRuleSupport: Double,
    removeFeatureGroup: Boolean,
    protectTextSharedHash: Boolean,
    removedGroups: Seq[String]
  ): List[String] = {
    if (isLabel) List() // never remove the label!
    else {

      val exclusionReasons = List(
        Option(variance).filter(_ <= minVariance).map(variance =>
          s"variance $variance lower than min variance $minVariance"
        ),
        corrLabel.filter(Math.abs(_) < minCorrelation).map(corr =>
          s"correlation $corr lower than min correlation $minCorrelation"
        ),
        corrLabel.filter(Math.abs(_) > maxCorrelation).map(corr =>
          s"correlation $corr higher than max correlation $maxCorrelation"
        ),
        cramersV.filter(_ > maxCramersV).map(cv =>
          s"Cramer's V $cv higher than max Cramer's V $maxCramersV"
        ),
        maxRuleConfidences.zip(supports).collectFirst {
          case (conf, sup) if (conf > maxRuleConfidence && sup > minRequiredRuleSupport) =>
            s"Max association rule confidence $conf is above threshold of $maxRuleConfidence and support $sup is " +
            s"above the required support threshold of $minRequiredRuleSupport"
        },
        column.flatMap(_.indicatorGroup).filter(removedGroups.contains(_)).map(ig =>
          s"other feature in indicator group $ig flagged for removal via rule confidence checks"
        )
      ).flatten

      val parentExclusionReasons =
        if (removeFeatureGroup && (!column.forall(isTextSharedHash) || !protectTextSharedHash)) {
          List(
            parentCramersV.filter(_ > maxCramersV).map(cv =>
              s"Cramer's V $cv for something in parent feature set higher than max Cramer's V $maxCramersV"),
            parentCorr.filter(_ > maxCorrelation).map(corr =>
              s"correlation $corr for something in parent feature set higher than max correlation $maxCorrelation")
          ).flatten
        } else List.empty[String]

      exclusionReasons ++ parentExclusionReasons
    }
  }

  /**
   * Is column a shared hash feature that is derived from Text, TextArea, TextMap, or TextAreaMap
   *
   * @param metadata     metadata of column
   * @return
   */
  def isTextSharedHash(metadata: OpVectorColumnMetadata): Boolean = {
    val isDerivedFromText = metadata.hasParentOfType[Text] || metadata.hasParentOfType[TextArea] ||
      metadata.hasParentOfType[TextMap] || metadata.hasParentOfType[TextAreaMap]
    isDerivedFromText && metadata.indicatorGroup.isEmpty && metadata.indicatorValue.isEmpty
  }

  override def toString: String = {
    val description = if (isLabel) "Label" else s"Feature"
    s"$description $name has: " +
      s"samples = $count, mean = $mean, min = $min, max = $max, variance = $variance" +
      corrLabel.fold("") { corr => s"\n$description $name has $corr correlation with label" } +
      cramersV.fold("") { corr => s"\n$description $name has $corr cramersV with label" } +
      parentCramersV.fold("") { corr => s"\n$description $name has parent feature $corr cramersV with label" }
  }

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
}

