/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}


/**
 * Contains all names for sanity checker metadata
 */
case object SanityCheckerNames {
  val CorrelationsWLabel: String = "correlationsWithLabel"
  val CorrelationsWLabelIsNaN: String = "correlationsWithLabelIsNaN"
  val CorrelationType: String = "correlationType"
  val CategoricalStats: String = "categoricalStats"
  val CategoricalFeatures: String = "categoricalFeatures"
  val CramersVIsNaN: String = "cramersVIsNaN"
  val CramersV: String = "cramersV"
  val MutualInfo: String = "mutualInfo"
  val PointwiseMutualInfoAgainstLabel: String = "pointwiseMutualInfoAgainstLabel"
  val CountMatrix: String = "countMatrix"
  val Names: String = "names"
  val FeaturesIn: String = "features"
  val Values: String = "values"
  val FeaturesStatistics = "statistics"
  val Dropped = "featuresDropped"
  val Mean = "mean"
  val Max = "max"
  val Min = "min"
  val Count = "count"
  val SampleFraction = "sampleFraction"
  val NumNonZeros = "numNonZeros"
  val Variance = "variance"
  val NumNull = "number of nulls"
}

/**
 * Case class to convert to and from [[SanityChecker]] summary metadata
 *
 * @param correlationsWLabel      feature correlations with label
 * @param dropped                 features dropped for label leakage
 * @param featuresStatistics      stats on features
 * @param names                   names of features passed in
 * @param categoricalStats
 */
case class SanityCheckerSummary
(
  correlationsWLabel: Correlations,
  dropped: Seq[String],
  featuresStatistics: SummaryStatistics,
  names: Seq[String],
  categoricalStats: CategoricalStats
) {

  private[op] def this(
    stats: Array[ColumnStatistics],
    catStats: CategoricalStats,
    dropped: Seq[String],
    colStats: MultivariateStatisticalSummary,
    names: Seq[String],
    correlationType: CorrelationType,
    sample: Double
  ) {
    this(
      correlationsWLabel = new Correlations(
        stats.filter(s => s.corrLabel.isDefined && !s.corrLabel.get.isNaN).map(s => s.name -> s.corrLabel.get),
        stats.filter(s => s.corrLabel.isDefined && s.corrLabel.get.isNaN).map(_.name),
        correlationType
      ),
      dropped = dropped,
      featuresStatistics = new SummaryStatistics(colStats, stats.map(_.numNulls), sample),
      names = names,
      categoricalStats = catStats
    )
  }

  /**
   * Convert to metadata instance
   *
   * @return
   */
  def toMetadata(): Metadata = {
    val summaryMeta = new MetadataBuilder()
    summaryMeta.putMetadata(SanityCheckerNames.CorrelationsWLabel, correlationsWLabel.toMetadata())
    summaryMeta.putStringArray(SanityCheckerNames.Dropped, dropped.toArray)
    summaryMeta.putMetadata(SanityCheckerNames.FeaturesStatistics, featuresStatistics.toMetadata())
    summaryMeta.putStringArray(SanityCheckerNames.Names, names.toArray)
    summaryMeta.putMetadata(SanityCheckerNames.CategoricalStats, categoricalStats.toMetadata())
    summaryMeta.build()
  }

}

/**
 * Statistics on features (zip arrays with names in SanityCheckerSummary to get feature associated with values)
 *
 * @param count          count of data in sample used to calculate stats
 * @param sampleFraction fraction of total data used in calculation
 * @param max            max value seen
 * @param min            min value
 * @param mean           mean value
 * @param variance       variance of value
 */
case class SummaryStatistics
(
  count: Double,
  sampleFraction: Double,
  max: Seq[Double],
  min: Seq[Double],
  mean: Seq[Double],
  variance: Seq[Double],
  numNull: Seq[Double]
) {

  private[op] def this(colStats: MultivariateStatisticalSummary, trackNulls: Array[Double], sample: Double) = this(
    count = colStats.count,
    sampleFraction = sample,
    max = colStats.max.toArray,
    min = colStats.min.toArray,
    mean = colStats.mean.toArray,
    variance = colStats.variance.toArray,
    numNull = trackNulls
  )

  /**
   * Convert to metadata instance
   *
   * @return
   */
  def toMetadata(): Metadata = {
    val meta = new MetadataBuilder()
    meta.putDouble(SanityCheckerNames.Count, count)
    meta.putDouble(SanityCheckerNames.SampleFraction, sampleFraction)
    meta.putDoubleArray(SanityCheckerNames.Max, max.toArray)
    meta.putDoubleArray(SanityCheckerNames.Min, min.toArray)
    meta.putDoubleArray(SanityCheckerNames.Mean, mean.toArray)
    meta.putDoubleArray(SanityCheckerNames.Variance, variance.toArray)
    meta.putDoubleArray(SanityCheckerNames.NumNull, numNull.toArray)
    meta.build()
  }

}

/**
 * Container class for statistics calculated from contingency tables constructed from categorical variables
 *
 * @param categoricalFeatures  Names of features that we performed categorical tests on
 * @param cramersVs            Values of cramersV for each feature
 *                             (should be the same for everything coming from the same contingency matrix)
 * @param pointwiseMutualInfos Map from label value (as a string) to an Array (over features) of PMI values
 * @param mutualInfos          Values of MI for each feature (should be the same for everything coming from the same
 *                             contingency matrix)
 * @param counts               Counts of occurance for categoricals (n x m array of arrays where n = number of labels
 *                             and m = number of features + 1 with last element being occurance count of labels
 */
case class CategoricalStats
(
  categoricalFeatures: Array[String] = Array.empty,
  cramersVs: Array[Double] = Array.empty,
  pointwiseMutualInfos: CategoricalStats.PointwiseMutualInfos.Type = CategoricalStats.PointwiseMutualInfos.empty,
  mutualInfos: Array[Double] = Array.empty,
  counts: CategoricalStats.PointwiseMutualInfos.Type = CategoricalStats.PointwiseMutualInfos.empty
) {
  // TODO: Build the metadata here instead of by treating Cramer's V and mutual info as correlations
  def toMetadata(): Metadata = {
    val meta = new MetadataBuilder()
    meta.putStringArray(SanityCheckerNames.CategoricalFeatures, categoricalFeatures)
    // TODO: use custom serializer here instead of replacing NaNs with 0s
    meta.putDoubleArray(SanityCheckerNames.CramersV, cramersVs.map(f => if (f.isNaN) 0 else f))
    meta.putDoubleArray(SanityCheckerNames.MutualInfo, mutualInfos)
    meta.putMetadata(SanityCheckerNames.PointwiseMutualInfoAgainstLabel, pointwiseMutualInfos.toMetadata)
    val countMeta = new MetadataBuilder()
    counts.map{ case (k, v) => countMeta.putDoubleArray(k, v)}
    meta.putMetadata(SanityCheckerNames.CountMatrix, countMeta.build())
    meta.build()
  }
}

object CategoricalStats {
  /**
   * Pointwise mutual information (PMI) values:
   * Map from label value (as a string) to an Array (over features) of PMI values
   */
  object PointwiseMutualInfos {
    type Type = Map[String, Array[Double]]
    def empty: Type = Map.empty
  }
}

/**
 * Correlations between features and the label from [[SanityChecker]]
 *
 * @param featuresIn names of features
 * @param values     correlation of feature with label
 * @param nanCorrs   nan correlation features
 * @param corrType   type of correlation done on
 */
case class Correlations
(
  featuresIn: Seq[String],
  values: Seq[Double],
  nanCorrs: Seq[String],
  corrType: CorrelationType
) {
  assert(featuresIn.length == values.length, "Feature names and correlation values arrays must have the same length")

  def this(corrs: Seq[(String, Double)], nans: Seq[String], corrType: CorrelationType) = this(
    featuresIn = corrs.map(_._1),
    values = corrs.map(_._2),
    nanCorrs = nans,
    corrType = corrType
  )

  /**
   * Convert to metadata instance
   *
   * @return
   */
  def toMetadata(): Metadata = {
    val corrMeta = new MetadataBuilder()
    corrMeta.putStringArray(SanityCheckerNames.FeaturesIn, featuresIn.toArray)
    corrMeta.putDoubleArray(SanityCheckerNames.Values, values.toArray)
    corrMeta.putStringArray(SanityCheckerNames.CorrelationsWLabelIsNaN, nanCorrs.toArray)
    corrMeta.putString(SanityCheckerNames.CorrelationType, corrType.name)
    corrMeta.build()
  }
}

case object SanityCheckerSummary {

  private def correlationsFromMetadata(meta: Metadata): Correlations = {
    val wrapped = meta.wrapped
    Correlations(
      featuresIn = wrapped.getArray[String](SanityCheckerNames.FeaturesIn).toSeq,
      values = wrapped.getArray[Double](SanityCheckerNames.Values).toSeq,
      nanCorrs = wrapped.getArray[String](SanityCheckerNames.CorrelationsWLabelIsNaN).toSeq,
      corrType = CorrelationType.withNameInsensitive(wrapped.get[String](SanityCheckerNames.CorrelationType))
    )
  }

  private def statisticsFromMetadata(meta: Metadata): SummaryStatistics = {
    val wrapped = meta.wrapped
    SummaryStatistics(
      count = wrapped.get[Double](SanityCheckerNames.Count),
      sampleFraction = wrapped.get[Double](SanityCheckerNames.SampleFraction),
      max = wrapped.getArray[Double](SanityCheckerNames.Max).toSeq,
      min = wrapped.getArray[Double](SanityCheckerNames.Min).toSeq,
      mean = wrapped.getArray[Double](SanityCheckerNames.Mean).toSeq,
      variance = wrapped.getArray[Double](SanityCheckerNames.Variance).toSeq,
      numNull = wrapped.getArray[Double](SanityCheckerNames.NumNull).toSeq
    )
  }

  private def categoricalStatsFromMetadata(meta: Metadata): CategoricalStats = {
    val wrapped = meta.wrapped
    CategoricalStats(
      categoricalFeatures = wrapped.getArray[String](SanityCheckerNames.CategoricalFeatures),
      cramersVs = wrapped.getArray[Double](SanityCheckerNames.CramersV),
      mutualInfos = wrapped.getArray[Double](SanityCheckerNames.MutualInfo),
      pointwiseMutualInfos = meta.getMetadata(SanityCheckerNames.PointwiseMutualInfoAgainstLabel)
        .underlyingMap.asInstanceOf[CategoricalStats.PointwiseMutualInfos.Type],
      counts = meta.getMetadata(SanityCheckerNames.CountMatrix)
        .underlyingMap.asInstanceOf[CategoricalStats.PointwiseMutualInfos.Type]
    )
  }

  /**
   * Converts metadata into instance of SanityCheckerSummary
   *
   * @param meta metadata produced by [[SanityChecker]] which contains summary information
   * @return an instance of the [[SanityCheckerSummary]]
   */
  def fromMetadata(meta: Metadata): SanityCheckerSummary = {
    val wrapped = meta.wrapped
    SanityCheckerSummary(
      correlationsWLabel = correlationsFromMetadata(wrapped.get[Metadata](SanityCheckerNames.CorrelationsWLabel)),
      dropped = wrapped.getArray[String](SanityCheckerNames.Dropped).toSeq,
      featuresStatistics = statisticsFromMetadata(wrapped.get[Metadata](SanityCheckerNames.FeaturesStatistics)),
      names = wrapped.getArray[String](SanityCheckerNames.Names).toSeq,
      categoricalStats = categoricalStatsFromMetadata(wrapped.get[Metadata](SanityCheckerNames.CategoricalStats))
    )
  }

}
