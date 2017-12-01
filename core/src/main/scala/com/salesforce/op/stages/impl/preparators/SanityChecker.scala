/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.preparators

import scala.math.min
import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.impl.preparators.CorrelationType.Pearson
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.stats.OpStatistics
import enumeratum._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors => NewVectors}
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder}

import scala.reflect.runtime.universe._
import scala.util.Try


trait SanityCheckerParams extends Params {
  final lazy val sampleLimit = new IntParam(
    parent = this, name = "sampleLimit",
    doc = "Upper limit on number of samples in downsampled data set (note: sample limit will not be exact, due " +
      "to Spark's dataset sampling behavior)",
    isValid = ParamValidators.gtEq(lowerBound = 0)
  )

  def setSampleLimit(value: Int): this.type = set(sampleLimit, value)

  def getSampleLimit: Int = $(sampleLimit)

  final lazy val checkSample = new DoubleParam(
    parent = this, name = "checkSample",
    doc = "Rate to downsample the data for statistical calculations (note: actual sampling will not be exact " +
      "due to Spark's dataset sampling behavior)",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = false, upperInclusive = true)
  )

  def setCheckSample(value: Double): this.type = set(checkSample, value)

  def getCheckSample: Double = $(checkSample)

  final lazy val sampleSeed = new LongParam(
    parent = this, name = "sampleSeed",
    doc = "Seed to use when sampling"
  )

  def setSampleSeed(value: Long): this.type = set(sampleSeed, value)

  def getSampleSeed: Long = $(sampleSeed)

  final lazy val removeBadFeatures = new BooleanParam(
    parent = this, name = "removeBadFeatures",
    doc = "If set to true, this will automatically remove all the bad features from the feature vector"
  )

  def setRemoveBadFeatures(value: Boolean): this.type = set(removeBadFeatures, value)

  def getRemoveBadFeatures: Boolean = $(removeBadFeatures)

  final lazy val maxCorrelation = new DoubleParam(
    parent = this, name = "maxCorrelation",
    doc = "Maximum correlation (absolute value) allowed between a feature in the feature vector and the label",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = true, upperInclusive = true)
  )

  def setMaxCorrelation(value: Double): this.type = set(maxCorrelation, value)

  def getMaxCorrelation: Double = $(maxCorrelation)

  final lazy val minCorrelation = new DoubleParam(
    parent = this, name = "minCorrelation",
    doc = "Minimum correlation (absolute value) allowed between a feature in the feature vector and the label",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = true, upperInclusive = true)
  )

  def setMinCorrelation(value: Double): this.type = set(minCorrelation, value)

  def getMinCorrelation: Double = $(minCorrelation)

  final lazy val correlationType = new Param[CorrelationType](
    parent = this, name = "correlationType",
    doc = "Which coefficient to use for computing correlation"
  )

  def setCorrelationType(value: CorrelationType): this.type = set(correlationType, value)

  def getCorrelationType: CorrelationType = $(correlationType)

  final lazy val minVariance = new DoubleParam(
    parent = this, name = "minVariance",
    doc = "Minimum amount of variance allowed for each feature and label"
  )

  def setMinVariance(value: Double): this.type = set(minVariance, value)

  def getMinVariance: Double = $(minVariance)

  final lazy val maxCramersV = new DoubleParam(
    parent = this, name = "maxCramersV",
    doc = "Maximum categorical correlation value (Cramer's V) allowed between a categorical feature in the" +
      "feature vector and a categorical label",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = true, upperInclusive = true)
  )

  def setMaxCramersV(value: Double): this.type = set(maxCramersV, value)

  def getMaxCramersV: Double = $(maxCramersV)

  final lazy val categoricalLabel = new BooleanParam(
    parent = this, name = "categoricalLabel",
    doc = "If true, then label is treated as categorical (eg. Cramer's V will be calculated between it and " +
      "categorical features). If this is not set, then use a max class fraction of 0.1 to estimate whether label is" +
      "categorical or not."
  )

  def setCategoricalLabel(value: Boolean): this.type = set(categoricalLabel, value)

  def getCategoricalLabel: Boolean = $(categoricalLabel)

  setDefault(
    checkSample -> SanityChecker.CheckSample,
    sampleSeed -> SanityChecker.SampleSeed,
    sampleLimit -> SanityChecker.SampleLimit,
    maxCorrelation -> SanityChecker.MaxCorrelation,
    minCorrelation -> SanityChecker.MinCorrelation,
    minVariance -> SanityChecker.MinVariance,
    maxCramersV -> SanityChecker.MaxCramersV,
    removeBadFeatures -> SanityChecker.RemoveBadFeatures,
    correlationType -> SanityChecker.CorrelationType
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
  ) with SanityCheckerParams {

  /**
   * Function to be called on setInput
   */
  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  override protected def outputIsResponse: Boolean = false

  private def makeColumnStatistics(
    metaCols: Seq[OpVectorColumnMetadata],
    labelColumnIndex: Int,
    corrMatrix: Matrix,
    statsSummary: MultivariateStatisticalSummary,
    categoricalStats: CategoricalStats
  ): Array[ColumnStatistics] = {
    // precompute all statistics to avoid rebuilding the vectors every time
    val means = statsSummary.mean
    val maxs = statsSummary.max
    val mins = statsSummary.min
    val count = statsSummary.count
    val variances = statsSummary.variance
    val cramersVMap = categoricalStats.categoricalFeatures.zip(categoricalStats.cramersVs).toMap[String, Double]

    // build map from indicator group name to any null indicator it may have
    val nullIndicators: Map[String, (OpVectorColumnMetadata, Int)] = {
      val nullGroups = for {
        col <- metaCols
        if col.isNullIndicator
        indicatorGroup <- col.indicatorGroup
      } yield (indicatorGroup, (col, col.index))

      nullGroups.groupBy(_._1).foreach {
        case (group, cols) =>
          require(cols.length == 1, s"Vector column $group has multiple null indicator fields: $cols")
      }
      nullGroups.toMap
    }

    def numNulls(col: OpVectorColumnMetadata) = nullIndicators.get(col.makeColName()).fold(0.0) {
      case (_, i) => means(i) * count
    }

    val parentMap = metaCols.flatMap(c => c.parentFeatureName.map(p => c.makeColName() -> p)).toMap

    val corrParent = metaCols.flatMap(c => c.parentFeatureName.map(_ -> corrMatrix(c.index, labelColumnIndex)))
      .groupBy(_._1).map(_._2.maxBy(_._2))

    val cramersVParent = cramersVMap.toSeq
      .map { case (k, v) => parentMap(k) -> v }.groupBy(_._1).map(_._2.maxBy(_._2))

    val featuresStats = metaCols.map {
      col =>
        val i = col.index
        ColumnStatistics(
          name = col.makeColName(),
          column = Some(col),
          isLabel = false,
          count = count,
          mean = means(i),
          min = mins(i),
          max = maxs(i),
          variance = variances(i),
          numNulls = numNulls(col),
          corrLabel = Option(corrMatrix(i, labelColumnIndex)),
          // TODO: Remove categorical stuff from ColumnStatistics or refactor
          cramersV = cramersVMap.get(col.makeColName()),
          parentCorr = col.parentFeatureName.map(corrParent.get).max,
          parentCramersV = col.parentFeatureName.map(cramersVParent.get).max
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
      numNulls = 0.0,
      corrLabel = None,
      cramersV = None,
      parentCorr = None,
      parentCramersV = None
    )
    (labelStats +: featuresStats).toArray
  }

  private def getFeaturesToDrop(stats: Array[ColumnStatistics]): Array[ColumnStatistics] = {
    val minVar = $(minVariance)
    val minCorr = $(minCorrelation)
    val maxCorr = $(maxCorrelation)
    val maxCramV = $(maxCramersV)
    for {
      col <- stats
      reasons = col.reasonsToRemove(
        minVariance = minVar,
        minCorrelation = minCorr,
        maxCorrelation = maxCorr,
        maxCramersV = maxCramV
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
    data: Dataset[(RealNN#Value, OPVector#Value)]
  ): CategoricalStats = {
    implicit val encDoubleArray: Encoder[Array[Double]] = ExpressionEncoder()

    val contingencyDataset = data.groupByKey(_._1).mapValues {
      // Add in the label with the features so we can get the number of positive responses in the same reduction
      case (label, features) => features.toArray :+ 1.0
    }.reduceGroups((a, b) => a.zip(b).map(f => f._1 + f._2))

    // Only calculate this if the label is categorical. Either the user specifies the label is categorical with
    // the categoricalLabel param, or if that is not set we assume the label is categorical if the number
    // of distinct labels is less than the min of 100 and sample size * 0.1
    val distictLabel = contingencyDataset.count()
    if (isDefined(categoricalLabel) && $(categoricalLabel) || distictLabel < min(100.0, sampleSize * 0.1)) {

      val contingencyWithKeys = contingencyDataset.collect()
      val contingency = contingencyWithKeys.sortBy(_._1).map { case (realNN, vec) => vec }

      logInfo("Label is assumed to be categorical since either categoricalLabel = true or " +
        "number of distinct labels < count * 0.1")

      // Only perform Cramer's V calculation on columns that have an indicatorGroup and indicatorValue defined (right
      // now, the only things that will have indicatorValue defined and indicatorGroup be None is numeric maps)
      val colIndicesByIndicatorGroup = columnMeta.filter { f =>
        val ftt = FeatureType.featureTypeTag(f.parentFeatureType.head).tpe
        f.indicatorGroup.isDefined && f.indicatorValue.isDefined &&
          !(ftt =:= weakTypeOf[MultiPickList] || ftt <:< weakTypeOf[OPMap[_]])
      }
        .map { meta => (meta.indicatorGroup.get, meta) }
        .groupBy(_._1)
        .map {
          case (group, cols) => (group, cols.map(_._2.makeColName()), cols.map(_._2.index))
        }

      val statsTuple = colIndicesByIndicatorGroup.flatMap {
        case (group, colNames, valueIndices) =>
          val groupContingency =
            if (valueIndices.length == 1) {
              // parentFeatureNames only has a single indicator column, construct the other from label sums
              contingency.flatMap(features => {
                val indicatorSum = valueIndices.map(features.apply)
                indicatorSum ++ indicatorSum.map(features.last - _)
              })
            } else contingency.flatMap { features => valueIndices.map(features.apply) }

          // columns are label value, rows are feature value
          val cStats =
            if (valueIndices.length == 1) {
              val contingencyMatrix = new DenseMatrix(2, groupContingency.length / 2, groupContingency)
              OpStatistics.contingencyStatsFromSingleColumn(contingencyMatrix)
            } else {
              val contingencyMatrix = new DenseMatrix(valueIndices.length,
                groupContingency.length / valueIndices.length, groupContingency)
              OpStatistics.contingencyStats(contingencyMatrix)
            }

          // Return all of the results in a single tuple, so we don't have to iterate through and construct the
          // contingency matrices again
          colNames.map(c => (c, group, cStats.cramersV, cStats.pointwiseMutualInfo, cStats.mutualInfo))
      }

      val statsTuplePerGroup = statsTuple.groupBy(_._2).map { case (k, v) => v.head }
      val reducedPMI = statsTuplePerGroup.map(_._4).foldLeft(Map.empty[String, Array[Double]]) {
        case (l, r) => r.map { case (k, v) => k.toString -> (l.getOrElse(k.toString, Array.empty[Double]) ++ v) }
      }
      CategoricalStats(
        categoricalFeatures = statsTuple.map(_._1).toArray,
        cramersVs = statsTuple.map(_._3).toArray,
        pointwiseMutualInfos = reducedPMI,
        mutualInfos = statsTuple.map(_._5).toArray
      )
    } else {
      logInfo(s"Label is assumed to be continuous since number of distinct labels = $distictLabel" +
        s"which is greater than 10% the size of the sample $sampleSize skipping calculation of Cramer's V")
      CategoricalStats()
    }
  }

  /**
   * The SanityChecker's core 'estimator' function, computes statistics on the features and the
   * list of features to be removed.
   */
  override def fitFn(data: Dataset[(RealNN#Value, OPVector#Value)]): BinaryModel[RealNN, OPVector, OPVector] = {
    data.persist()
    val ckSample = $(checkSample)
    val sampSeed = $(sampleSeed)
    val removeBad = $(removeBadFeatures)
    val corrType = $(correlationType)
    val sampleData = if (ckSample >= 1.0) {
      logInfo(s"NOT sampling the data for Sanity Checker, since the specified check sample is $ckSample")
      data
    }
    else {
      logInfo(s"Sampling the data for Sanity Checker with sample $ckSample and seed $sampSeed")
      val dataCount = data.count()
      data.sample(
        withReplacement = false,
        fraction = math.min(ckSample, $(sampleLimit).toDouble / dataCount),
        seed = sampSeed
      )
    }

    implicit val enc: Encoder[OldVector] = ExpressionEncoder()
    logInfo("Getting vector rows")
    val vectorRows: Dataset[OldVector] = sampleData.map {
      case (Some(label), sparse: SparseVector) =>
        val newSize = sparse.size + 1
        if (label != 0.0) OldVectors.sparse(newSize, sparse.indices :+ sparse.size, sparse.values :+ label)
        else OldVectors.sparse(newSize, sparse.indices, sparse.values)
      case (Some(label), dense: DenseVector) =>
        OldVectors.dense(dense.toArray :+ label)
      case (label, _) => throw new IllegalArgumentException("Sanity checker input missing label for row")
    }.persist()

    logInfo("Calculating columns stats")
    val colStats = Statistics.colStats(vectorRows.rdd)

    // do this instead of vectorRows.first().size - 1 to avoid having to submit a job for the spark action
    val featureSize = colStats.mean.size - 1
    val labelColumn = featureSize // label column goes at end of vector

    // handle any possible serialization errors if users give us wrong metadata
    val vectorMeta = Try {
      OpVectorMetadata(getInputSchema()(in2.name))
    }.recover {
      case e: NoSuchElementException =>
        throw new IllegalArgumentException("Vector input metadata is malformed: ", e)
    }.get

    require(featureSize == vectorMeta.size,
      "Number of columns in vector metadata did not match number of columns in data, check your vectorizers")
    val vectorMetaColumns = vectorMeta.columns
    val featureNames = vectorMetaColumns.map(_.makeColName())
    require(featureSize > 0, "Feature vector passed in is empty, check your vectorizers")

    logInfo(s"Calculating ${corrType.name} correlations")
    val covariance = Statistics.corr(vectorRows.rdd, corrType.name)

    // Only calculate this if the label is categorical, so ignore if user has flagged label as not categorical
    val categoricalStats =
      if (isDefined(categoricalLabel) && !$(categoricalLabel)) {
        CategoricalStats()
      } else {
        logInfo("Attempting to calculate Cramer's V between each categorical feature and the label")
        categoricalTests(colStats.count, featureSize, vectorMetaColumns, sampleData)
      }

    logInfo("Logging all statistics")
    val stats = makeColumnStatistics(vectorMetaColumns, labelColumn, covariance, colStats, categoricalStats)
    require(colStats.count > 0, "Nothing has been added to this colStats")
    stats.foreach { stat => logInfo(stat.toString) }

    logInfo("Calculating features to remove")
    val toDropFeatures = if (removeBad) getFeaturesToDrop(stats) else Array.empty[ColumnStatistics]
    val toDropSet = toDropFeatures.flatMap(_.column).toSet
    val outputFeatures = vectorMetaColumns.filterNot { col => toDropSet.contains(col) }
    val indicesToKeep = outputFeatures.map(_.index)

    val outputMeta = OpVectorMetadata(in2.name, outputFeatures, vectorMeta.history)

    // TODO: Refactor so that everything is constructed directly from our Array[ColumnStatistics], stats
    val summary = new SanityCheckerSummary(
      stats = stats,
      catStats = categoricalStats,
      dropped = toDropFeatures.map(_.name),
      colStats = colStats,
      names = featureNames :+ in1.name,
      correlationType = corrType,
      sample = ckSample
    )
    setMetadata(outputMeta.toMetadata.withSummaryMetadata(summary.toMetadata()))

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

private final class SanityCheckerModel
(
  indicesToKeep: Array[Int],
  removeBadFeatures: Boolean,
  operationName: String,
  uid: String
) extends BinaryModel[RealNN, OPVector, OPVector](operationName = operationName, uid = uid) {

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
  val SampleLimit = 100000
  val MaxCorrelation = 0.95
  val MinCorrelation = 0.0
  val MinVariance = 0.00001
  val MaxCramersV = 0.95
  val RemoveBadFeatures = false
  val CorrelationType = Pearson

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
  numNulls: Double,
  corrLabel: Option[Double],
  cramersV: Option[Double],
  parentCorr: Option[Double],
  parentCramersV: Option[Double]
) {

  /**
   * Given a minimum variance, maximum variance, and maximum correlation, decide if there is a reason to remove
   * this column. If so, return a list of the reasons why. If not, then return an empty list.
   *
   * @param minVariance    Minimum variance
   * @param maxCorrelation Maximum correlation
   * @param minCorrelation Minimum correlation
   * @param maxCramersV    Maximum Cramer's V value
   * @return List[String] if reason to remove, nil otherwise
   */
  def reasonsToRemove(
    minVariance: Double,
    maxCorrelation: Double,
    minCorrelation: Double,
    maxCramersV: Double
  ): List[String] = {
    if (isLabel) List() // never remove the label!
    else List(
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
      parentCramersV.filter(_ > maxCramersV).map(cv =>
        s"Cramer's V $cv for something in parent feature set higher than max Cramer's V $maxCramersV"
      )
    ).flatten
  }

  override def toString: String = {
    val description = if (isLabel) "Label" else s"Feature"
    s"$description $name has: " +
      s"samples = $count, mean = $mean, min = $min, max = $max, variance = $variance, number of nulls = $numNulls" +
      corrLabel.fold("") { corr => s"\n$description $name has $corr correlation with label" } +
      cramersV.fold("") { corr => s"\n$description $name has $corr cramersV with label" } +
      parentCramersV.fold("") { corr => s"\n$description $name has parent feature $corr cramersV with label" }
  }

}

/**
 * Represents a kind of correlation coefficient.
 *
 * @param name The spark name of the correlation coefficient
 */
sealed abstract class CorrelationType(val name: String) extends EnumEntry with Serializable

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
