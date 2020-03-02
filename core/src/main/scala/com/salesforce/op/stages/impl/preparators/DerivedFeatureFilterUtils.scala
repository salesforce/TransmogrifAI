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

import com.salesforce.op.features.types.{OPVector, Text, TextArea, TextAreaMap, TextMap, VectorConversions}
import com.salesforce.op.utils.spark.OpVectorColumnMetadata
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.log4j.Level
import org.apache.spark.ml.linalg.{Vectors => NewVectors}
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, Param, Params}
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.sql.types.Metadata

import scala.util.Try


trait DerivedFeatureFilterParams extends Params {

  final val logLevel = new Param[String](
    parent = this, name = "logLevel",
    doc = "sets log level (INFO, WARN, ERROR, DEBUG etc.)",
    isValid = (s: String) => Try(Level.toLevel(s)).isSuccess
  )

  private[op] def setLogLevel(level: Level): this.type = set(logLevel, level.toString)

  final val removeBadFeatures = new BooleanParam(
    parent = this, name = "removeBadFeatures",
    doc = "If set to true, this will automatically remove all the bad features from the feature vector"
  )

  def setRemoveBadFeatures(value: Boolean): this.type = set(removeBadFeatures, value)

  def getRemoveBadFeatures: Boolean = $(removeBadFeatures)

  final val minVariance = new DoubleParam(
    parent = this, name = "minVariance",
    doc = "Minimum amount of variance allowed for each feature"
  )

  def setMinVariance(value: Double): this.type = set(minVariance, value)

  def getMinVariance: Double = $(minVariance)

  setDefault(
    removeBadFeatures -> DerivedFeatureFilter.RemoveBadFeatures,
    minVariance -> DerivedFeatureFilter.MinVariance
  )
}

object DerivedFeatureFilterUtils {

  /**
   * Builds an Array of ColumnStatistics objects containing all the data we calculate for each column (eg. mean,
   * max, variance, correlation, cramer's V, etc.)
   *
   * @param metaCols          Sequence of OpVectorColumnMetadata to use for grouping features
   * @param statsSummary      Multivariate statistics previously computed by Spark
   * @param labelNameAndIndex Name of label and index of the column corresponding to the label
   * @param corrsWithLabel    Array containing correlations between each feature vector element and the label
   * @param corrIndices       Indices that we actually compute correlations for (eg. can ignore hashed text features)
   * @param categoricalStats  Array of CategoricalGroupStats for each group of feature vector indices corresponding
   *                          to a categorical feature
   * @return Array of ColumnStatistics objects, one for each column in `metaCols`
   */
  def makeColumnStatistics
  (
    metaCols: Seq[OpVectorColumnMetadata],
    statsSummary: MultivariateStatisticalSummary,
    labelNameAndIndex: Option[(String, Int)] = None,
    corrsWithLabel: Array[Double] = Array.empty,
    corrIndices: Array[Int] = Array.empty,
    categoricalStats: Array[CategoricalGroupStats] = Array.empty
  ): Array[ColumnStatistics] = {
    // precompute all statistics to avoid rebuilding the vectors every time
    val means = statsSummary.mean
    val maxs = statsSummary.max
    val mins = statsSummary.min
    val count = statsSummary.count
    val variances = statsSummary.variance
    val cramersVMap = categoricalStats.flatMap(f => f.categoricalFeatures.map(c => c -> f.cramersV))
      .toMap[String, Double]
    val numCorrIndices = corrIndices.length

    def maxByParent(seq: Seq[(String, Double)]) = seq.groupBy(_._1).map { case (k, v) =>
      // Filter out the NaNs because max(3.4, NaN) = NaN, and we still want the keep the largest correlation
      k -> v.filterNot(_._2.isNaN).foldLeft(0.0)((a, b) => math.max(a, math.abs(b._2)))
    }

    def corrParentMap(fn: OpVectorColumnMetadata => Seq[String]) =
      maxByParent(metaCols.flatMap(c =>
        // Need to map feature indices to indices in correlation matrix, since we might skip hashed text indices
        corrIndices.indexOf(c.index) match {
          case -1 => Seq.empty[(String, Double)]
          case i => fn(c).map(_ -> corrsWithLabel(i))
        }
      ))

    def cramersVParentMap(fn: OpVectorColumnMetadata => Seq[String]) = {
      val parentMap = metaCols.flatMap { c => fn(c).map(c.makeColName() -> _) }.toMap
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
      col.parentNamesWithMapKeys().flatMap(k => check1.get(k).orElse(check2.get(k))).reduceOption(_ max _)

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
          // Label index is always the last index, which depends on how many indices we calculate correlations for
          corrLabel = corrIndices.indexOf(i) match {
            case -1 => None
            case ind => Option(corrsWithLabel(ind))
          },
          cramersV = cramersVMap.get(name),
          parentCorr = getParentValue(col, corrParent, corrParentNoKeys),
          parentCramersV = getParentValue(col, cramersVParent, cramersVParentNoKeys),
          maxRuleConfidences = maxRuleConfMap.getOrElse(name, Seq.empty),
          supports = supportMap.getOrElse(name, Seq.empty)
        )
    }

    val columnStatistics = labelNameAndIndex match {
      case None => featuresStats.toArray
      case Some((labelName: String, labelColumnIndex: Int)) => {
        val labelStats = ColumnStatistics(
          name = labelName,
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
    }
    columnStatistics
  }

  /**
   * Identifies which features to drop based on input exclusion criteria, and returns
   * array of dropped columns, with messages for logging why columns were dropped
   *
   * @param stats                  ColumnStatistics containing multivariate statistics computed by Spark
   * @param minVariance            Min variance for dropping features
   * @param minCorrelation         Min correlation with label for dropping features
   * @param maxCorrelation         Max correlation with label for dropping features
   * @param maxCramersV            Max Cramer's V for dropping categorical features
   * @param maxRuleConfidence      Max allowed confidence of association rules for dropping features
   * @param minRequiredRuleSupport Threshold for association rule
   * @param removeFeatureGroup     Whether to remove features descended from parent feature with derived features
   *                               that meet exclusion criteria
   * @param protectTextSharedHash  Whether individual hash is dropped or kept independently of related null
   *                               indicators or other hashes
   * @return columns to drop, with exclusion reasons
   */
  def getFeaturesToDrop
  (
    stats: Array[ColumnStatistics],
    minVariance: Double,
    minCorrelation: Double = 0.0,
    maxCorrelation: Double = 1.0,
    maxCramersV: Double = 1.0,
    maxRuleConfidence: Double = 1.0,
    minRequiredRuleSupport: Double = 1.0,
    removeFeatureGroup: Boolean = false,
    protectTextSharedHash: Boolean = true
  ): Array[(ColumnStatistics, String)] = {

    // Calculate groups to remove separately. This is for more complicated checks where you can't determine whether
    // to remove a feature from a single column stats (eg. associate rule confidence/support check)
    val groupByGroups = stats.groupBy(_.column.flatMap(_.featureGroup()))
    val ruleConfGroupsToDrop = groupByGroups.toSeq.flatMap {
      case (Some(group), colStats) =>
        val colsToRemove = colStats.filter(f =>
          f.maxRuleConfidences.zip(f.supports).exists {
            case (maxConf, sup) => (maxConf > maxRuleConfidence) && (sup > minRequiredRuleSupport)
          })
        if (colsToRemove.nonEmpty) Option(group) else None

      case _ => None
    }

    for {
      col <- stats
      reasons = col.reasonsToRemove(
        minVariance = minVariance,
        minCorrelation = minCorrelation,
        maxCorrelation = maxCorrelation,
        maxCramersV = maxCramersV,
        maxRuleConfidence = maxRuleConfidence,
        minRequiredRuleSupport = minRequiredRuleSupport,
        removeFeatureGroup = removeFeatureGroup,
        protectTextSharedHash = protectTextSharedHash,
        removedGroups = ruleConfGroupsToDrop
      )
      if reasons.nonEmpty
    } yield {
      val warning = s"Removing ${col.name} due to: ${reasons.mkString(",")}"
      (col, warning)
    }
  }

  /**
   * Transformation used in derived feature filters. If `removeBadFeatures` true, then this is just
   * identity (does nothing); otherwise, returns OPVector with only columns in `indicesToKeep`
   *
   * @param indicesToKeep     column indices of derived features to keep
   * @param removeBadFeatures whether to remove any features
   * @return [[OPVector]] with bad features dropped if `removeBadFeatures` true
   */
  def removeFeatures
  (
    indicesToKeep: Array[Int],
    removeBadFeatures: Boolean
  ): OPVector => OPVector = feature => {
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
  supports: Seq[Double],
  removalReasons: List[String] = List.empty[String]
) {

  /**
   * Given a minimum variance, maximum variance, and maximum correlation, decide if there is a reason to remove
   * this column. If so, return a list of the reasons why. If not, then return an empty list.
   *
   * @param minVariance            Minimum variance
   * @param maxCorrelation         Maximum correlation
   * @param minCorrelation         Minimum correlation
   * @param maxCramersV            Maximum Cramer's V value
   * @param maxRuleConfidence      Minimum association rule confidence between
   * @param minRequiredRuleSupport Minimum required support to throw away a group
   * @param removeFeatureGroup     Whether to remove entire feature group when any group value is flagged for removal
   * @param protectTextSharedHash  Whether to protect text shared hash from related null indicator and other hashes
   * @param removedGroups          Pre-determined feature groups to remove (eg. via maxRuleConfidence)
   * @return List[String] if reason to remove, nil otherwise
   */
  def reasonsToRemove
  (
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
        column.flatMap(_.featureGroup()).filter(removedGroups.contains(_)).map(ig =>
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
   * @param metadata metadata of column
   * @return
   */
  def isTextSharedHash(metadata: OpVectorColumnMetadata): Boolean = {
    val isDerivedFromText = metadata.hasParentOfType[Text] || metadata.hasParentOfType[TextArea] ||
      metadata.hasParentOfType[TextMap] || metadata.hasParentOfType[TextAreaMap]
    isDerivedFromText && metadata.grouping.isEmpty && metadata.indicatorValue.isEmpty
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

trait DerivedFeatureFilterNames {
  val FeaturesStatistics = "statistics"
  val Dropped = "featuresDropped"
  val Names: String = "names"
  val Count = "count"
  val SampleFraction = "sampleFraction"
  val Mean = "mean"
  val Max = "max"
  val Min = "min"
  val Variance = "variance"
}

trait DerivedFeatureFilterSummary extends DerivedFeatureFilterNames {
  def statisticsFromMetadata(meta: Metadata): SummaryStatistics = {
    val wrapped = meta.wrapped
    SummaryStatistics(
      count = wrapped.get[Double](Count),
      sampleFraction = wrapped.get[Double](SampleFraction),
      max = wrapped.getArray[Double](Max).toSeq,
      min = wrapped.getArray[Double](Min).toSeq,
      mean = wrapped.getArray[Double](Mean).toSeq,
      variance = wrapped.getArray[Double](Variance).toSeq
    )
  }
}

object DerivedFeatureFilter {
  val RemoveBadFeatures = false
  val MinVariance = 1E-5
}

