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

import com.salesforce.op.stages.impl.MetadataLike

import scala.util.{Failure, Success, Try}
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.stats.OpStatistics
import com.salesforce.op.utils.stats.OpStatistics.LabelWiseValues
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
  val Group: String = "group"
  val CramersVIsNaN: String = "cramersVIsNaN"
  val CramersV: String = "cramersV"
  val MutualInfo: String = "mutualInfo"
  val ContingencyMatrix: String = "contingencyMatrix"
  val PointwiseMutualInfoAgainstLabel: String = "pointwiseMutualInfoAgainstLabel"
  val MaxRuleConfidence: String = "maxRuleConfidence"
  val Support: String = "support"
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
  categoricalStats: Array[CategoricalGroupStats]
) extends MetadataLike {

  private[op] def this(
    stats: Array[ColumnStatistics],
    catStats: Array[CategoricalGroupStats],
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
      featuresStatistics = new SummaryStatistics(colStats, sample),
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
    summaryMeta.putMetadataArray(SanityCheckerNames.CategoricalStats, categoricalStats.map(_.toMetadata()))
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
  variance: Seq[Double]
) extends MetadataLike {

  private[op] def this(colStats: MultivariateStatisticalSummary, sample: Double) = this(
    count = colStats.count,
    sampleFraction = sample,
    max = colStats.max.toArray,
    min = colStats.min.toArray,
    mean = colStats.mean.toArray,
    variance = colStats.variance.toArray
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
    meta.build()
  }

}

/**
 * Container for categorical stats coming from a single group (and therefore a single contingency matrix)
 *
 * @param group               Indicator group for this contingency matrix
 * @param categoricalFeatures Array of categorical features belonging to this group
 * @param contingencyMatrix   Contingency matrix for this feature group
 * @param pointwiseMutualInfo Matrix of PMI values in Map form (label -> PMI values)
 * @param cramersV            Cramer's V value for this feature group (how strongly correlated is it with the label)
 * @param mutualInfo          Mutual info value for this feature group
 * @param maxRuleConfidences  Array (one value per contingency matrix row) containing the largest association rule
 *                            confidence for that row (over all the labels)
 * @param supports            Array (one value per contingency matrix row) containing the supports for each categorical
 *                            choice (fraction of dats in which it is chosen)
 */
case class CategoricalGroupStats
(
  group: String,
  categoricalFeatures: Array[String],
  contingencyMatrix: LabelWiseValues.Type,
  pointwiseMutualInfo: LabelWiseValues.Type,
  cramersV: Double,
  mutualInfo: Double,
  maxRuleConfidences: Array[Double],
  supports: Array[Double]
) extends MetadataLike {
  /**
   * @return metadata of this specific categorical group
   */
  def toMetadata(): Metadata = {
    val meta = new MetadataBuilder()
    meta.putString(SanityCheckerNames.Group, group)
    meta.putStringArray(SanityCheckerNames.CategoricalFeatures, categoricalFeatures)
    meta.putMetadata(SanityCheckerNames.ContingencyMatrix, contingencyMatrix.toMetadata)
    meta.putMetadata(SanityCheckerNames.PointwiseMutualInfoAgainstLabel, pointwiseMutualInfo.toMetadata)
    meta.putDouble(SanityCheckerNames.CramersV, if (cramersV.isNaN) 0 else cramersV)
    meta.putDouble(SanityCheckerNames.MutualInfo, mutualInfo)
    meta.putDoubleArray(SanityCheckerNames.MaxRuleConfidence, maxRuleConfidences)
    meta.putDoubleArray(SanityCheckerNames.Support, supports)
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
 * @param counts               Counts of occurrence for categoricals (n x m array of arrays where n = number of labels
 *                             and m = number of features + 1 with last element being occurrence count of labels
 */
@deprecated("Functionality replaced by Array[CategoricalGroupStats]", "3.3.0")
case class CategoricalStats
(
  categoricalFeatures: Array[String] = Array.empty,
  cramersVs: Array[Double] = Array.empty,
  pointwiseMutualInfos: LabelWiseValues.Type = LabelWiseValues.empty,
  mutualInfos: Array[Double] = Array.empty,
  counts: LabelWiseValues.Type = LabelWiseValues.empty
) extends MetadataLike {
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
) extends MetadataLike {
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
    corrMeta.putString(SanityCheckerNames.CorrelationType, corrType.sparkName)
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
      variance = wrapped.getArray[Double](SanityCheckerNames.Variance).toSeq
    )
  }

  private def categoricalGroupStatsFromMetadata(meta: Metadata): CategoricalGroupStats = {
    val wrapped = meta.wrapped
    CategoricalGroupStats(
      group = wrapped.get[String](SanityCheckerNames.Group),
      categoricalFeatures = wrapped.getArray[String](SanityCheckerNames.CategoricalFeatures),
      contingencyMatrix = meta.getMetadata(SanityCheckerNames.ContingencyMatrix)
        .underlyingMap.asInstanceOf[LabelWiseValues.Type],
      pointwiseMutualInfo = meta.getMetadata(SanityCheckerNames.PointwiseMutualInfoAgainstLabel)
        .underlyingMap.asInstanceOf[LabelWiseValues.Type],
      cramersV = wrapped.get[Double](SanityCheckerNames.CramersV),
      mutualInfo = wrapped.get[Double](SanityCheckerNames.MutualInfo),
      maxRuleConfidences = wrapped.getArray[Double](SanityCheckerNames.MaxRuleConfidence),
      supports = wrapped.getArray[Double](SanityCheckerNames.Support)
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
    // Try parsing as an older version of metadata (pre-3.3.0) if this doesn't work
    Try {
      SanityCheckerSummary(
        correlationsWLabel = correlationsFromMetadata(wrapped.get[Metadata](SanityCheckerNames.CorrelationsWLabel)),
        dropped = wrapped.getArray[String](SanityCheckerNames.Dropped).toSeq,
        featuresStatistics = statisticsFromMetadata(wrapped.get[Metadata](SanityCheckerNames.FeaturesStatistics)),
        names = wrapped.getArray[String](SanityCheckerNames.Names).toSeq,
        categoricalStats = wrapped.getArray[Metadata](SanityCheckerNames.CategoricalStats)
          .map(categoricalGroupStatsFromMetadata)
      )
    } match {
      case Success(summary) => summary
      // Parse it under the old format
      case Failure(_) => throw new IllegalArgumentException(s"failed to parse SanityCheckerSummary from $meta")
    }
  }
}
