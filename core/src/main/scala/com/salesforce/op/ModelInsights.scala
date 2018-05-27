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

package com.salesforce.op

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types.{OPVector, RealNN}
import com.salesforce.op.stages.impl.preparators._
import com.salesforce.op.stages.impl.selector.{ModelSelectorBase, SelectedModel}
import com.salesforce.op.stages.{OPStage, OpPipelineStageParams, OpPipelineStageParamsNames}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.{Model, PipelineStage, Transformer}
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization.{write, writePretty}
import org.slf4j.LoggerFactory

import scala.util.Try

/**
 * Summary of all model insights
 *
 * @param label             summary of information about the label
 * @param features          sequence containing insights for each raw feature that fed into the model
 * @param selectedModelInfo summary information about model training and winning model from model selector
 * @param trainingParams    op parameters used in model training
 * @param stageInfo         all stages and their parameters settings used to create feature output of model
 *                          keyed by stageName
 */
case class ModelInsights
(
  label: LabelSummary,
  features: Seq[FeatureInsights],
  selectedModelInfo: Map[String, Any],
  trainingParams: OpParams,
  stageInfo: Map[String, Any]
) {

  def toJson(pretty: Boolean = true): String = {
    implicit val formats = DefaultFormats
    if (pretty) writePretty(this) else write(this)
  }
}

/**
 * Summary information about label used in model creation (all fields will be empty if no label is found)
 *
 * @param labelName      name of label feature
 * @param rawFeatureName name of raw features that label is derived from
 * @param rawFeatureType types of raw features that label is derived from
 * @param stagesApplied  the stageNames of all stages applied to label before modeling
 * @param sampleSize     count of label used to compute distribution information
 *                       (will be fraction of data corresponding to sample rate in sanity checker)
 * @param distribution   summary of label distribution (either continuous or discrete)
 */
case class LabelSummary
(
  labelName: Option[String] = None,
  rawFeatureName: Seq[String] = Seq.empty,
  rawFeatureType: Seq[String] = Seq.empty,
  stagesApplied: Seq[String] = Seq.empty,
  sampleSize: Option[Double] = None,
  distribution: Option[LabelInfo] = None
)

/**
 * Common trait for Continuous and Discrete
 */
trait LabelInfo

/**
 * Summary of label distribution for continuous label
 *
 * @param min      min value
 * @param max      max value
 * @param mean     mean value
 * @param variance variance of values
 */
case class Continuous(min: Double, max: Double, mean: Double, variance: Double) extends LabelInfo

/**
 * Summary of label distribution for discrete label
 *
 * @param domain sequence of all unique values observed in data
 * @param prob   probabilities of each unique value observed in data (order is matched to domain order)
 */
case class Discrete(domain: Seq[String], prob: Seq[Double]) extends LabelInfo

/**
 * Summary of feature insights for all features derived from a given input (raw) feature
 *
 * @param featureName     name of raw feature insights are about
 * @param featureType     type of raw feature insights are about
 * @param derivedFeatures sequence containing insights for each feature derived from the raw feature
 */
case class FeatureInsights(featureName: String, featureType: String, derivedFeatures: Seq[Insights])

/**
 * Summary of insights for a derived feature
 *
 * @param derivedFeatureName         name of derived feature
 * @param stagesApplied              the stageNames of all stages applied to make feature from the raw input feature
 * @param derivedFeatureGroup        grouping of this feature if the feature is a pivot
 * @param derivedFeatureValue        value of the feature if the feature is a numeric encoding of a non-numeric feature
 *                                   or bucket
 * @param excluded                   was this derived feature excluded from the model by the sanity checker
 * @param corr                       the correlation of this feature with the label
 * @param cramersV                   the cramersV of this feature with the label
 *                                   (when both label and feature are categorical)
 * @param mutualInformation          the mutual information for this feature
 *                                   (and all features in its grouping) with the label
 *                                   (categorical features only)
 * @param pointwiseMutualInformation the mutual information of this feature with each value of the label
 *                                   (categorical features only)
 * @param countMatrix                the counts of the occurrence of this feature with each of the label values
 *                                   (categorical features only)
 * @param contribution               the contribution of this feature to the model
 *                                   (eg feature importance for random forest, weight for logistic regression)
 * @param min                        the min value of this feature
 * @param max                        the max value of this feature
 * @param mean                       the mean value of this feature
 * @param variance                   the variance of this feature
 */
case class Insights
(
  derivedFeatureName: String,
  stagesApplied: Seq[String],
  derivedFeatureGroup: Option[String],
  derivedFeatureValue: Option[String],
  excluded: Option[Boolean] = None,
  corr: Option[Double] = None,
  cramersV: Option[Double] = None,
  mutualInformation: Option[Double] = None,
  pointwiseMutualInformation: Map[String, Double] = Map.empty,
  countMatrix: Map[String, Double] = Map.empty,
  contribution: Seq[Double] = Seq.empty,
  min: Option[Double] = None,
  max: Option[Double] = None,
  mean: Option[Double] = None,
  variance: Option[Double] = None
)

case object ModelInsights {
  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  /**
   * Read ModelInsights from a json
   *
   * @param json model insights in json
   * @return Try[ModelInsights]
   */
  def fromJson(json: String): Try[ModelInsights] = Try {
    implicit val formats = DefaultFormats
    parse(json).extract[ModelInsights]
  }

  /**
   * Function to extract the model summary info from the stages used to create the selected model output feature
   *
   * @param stages         stages used to make the feature
   * @param rawFeatures    raw features in the workflow
   * @param trainingParams parameters used to create the workflow model
   * @return model insight summary
   */
  private[op] def extractFromStages(
    stages: Array[OPStage],
    rawFeatures: Array[features.OPFeature],
    trainingParams: OpParams,
    blacklistedFeatures: Array[features.OPFeature]
  ): ModelInsights = {
    val sanityCheckers = stages.collect { case s: SanityCheckerModel => s }
    val sanityChecker = sanityCheckers.lastOption
    val checkerSummary = sanityChecker.map(s => SanityCheckerSummary.fromMetadata(s.getMetadata().getSummaryMetadata()))
    log.info(
      s"Found ${sanityCheckers.length} sanity checkers will " +
        s"${sanityChecker.map("use results from the last checker:" + _.uid + "to").getOrElse("not")}" +
        s" to fill in model insights"
    )

    val models = stages.collect { case s: SelectedModel => s } // TODO support other model types?
    val model = models.lastOption
    log.info(
      s"Found ${models.length} models will " +
        s"${model.map("use results from the last model:" + _.uid + "to").getOrElse("not")}" +
        s" to fill in model insights"
    )

    val label = model.map(_.getInputFeature[RealNN](0)).orElse(sanityChecker.map(_.getInputFeature[RealNN](0))).flatten
    log.info(s"Found ${label.map(_.name + " as label").getOrElse("no label")} to fill in model insights")


    // Recover the vector metadata
    val vectorInput: Option[OpVectorMetadata] = {
      def makeMeta(s: => OpPipelineStageParams) = Try(OpVectorMetadata(s.getInputSchema().last)).toOption

      sanityChecker
        // first try out to get vector metadata from sanity checker
        .flatMap(s => makeMeta(s.parent.asInstanceOf[SanityChecker]).orElse(makeMeta(s)))
        // fall back to model selector stage metadata
        .orElse(model.flatMap(m => makeMeta(m.parent.asInstanceOf[ModelSelectorBase[_, _]])))
        // finally try to get it from the last vector stage
        .orElse(
        stages.filter(_.getOutput().isSubtypeOf[OPVector]).lastOption
          .map(v => OpVectorMetadata(v.getOutputFeatureName, v.getMetadata()))
      )
    }
    log.info(
      s"Found ${vectorInput.map(_.name + " as feature vector").getOrElse("no feature vector")}" +
        s" to fill in model insights"
    )
    ModelInsights(
      label = getLabelSummary(label, checkerSummary),
      features = getFeatureInsights(vectorInput, checkerSummary, model, rawFeatures, blacklistedFeatures),
      selectedModelInfo = getModelInfo(model),
      trainingParams = trainingParams,
      stageInfo = getStageInfo(stages)
    )
  }

  private[op] def getLabelSummary(
    label: Option[FeatureLike[RealNN]],
    summary: Option[SanityCheckerSummary]
  ): LabelSummary = {
    label match {
      case Some(l) =>
        val history = l.history()
        val raw = l.rawFeatures
        val sample = summary.map(_.featuresStatistics.count)
        val info: Option[LabelInfo] = summary.map { s =>
          if (s.categoricalStats.isEmpty) {
            Continuous(
              min = s.featuresStatistics.min.last,
              max = s.featuresStatistics.max.last,
              mean = s.featuresStatistics.mean.last,
              variance = s.featuresStatistics.variance.last
            )
          } else {
            // Can pick any contingency matrix to compute the label stats
            val labelCounts = s.categoricalStats.head.contingencyMatrix.map{
              case (k, v) => k -> v.sum
            }.toSeq.sortBy(_._1)
            val totalCount = labelCounts.foldLeft(0.0)((acc, el) => acc + el._2)

            Discrete(
              domain = labelCounts.map(_._1),
              prob = labelCounts.map(_._2 / totalCount)
            )
          }
        }
        LabelSummary(Option(l.name), history.originFeatures, raw.map(_.typeName), history.stages, sample, info)
      case None => LabelSummary()
    }
  }

  private[op] def getFeatureInsights(
    vectorInfo: Option[OpVectorMetadata],
    summary: Option[SanityCheckerSummary],
    model: Option[SelectedModel],
    rawFeatures: Array[features.OPFeature],
    blacklistedFeatures: Array[features.OPFeature]
  ): Seq[FeatureInsights] = {
    val contributions = getModelContributions(model)

    val featureInsights = (vectorInfo, summary) match {
      case (Some(v), Some(s)) =>
        val droppedSet = s.dropped.toSet
        val indexInToIndexKept = v.columns
          .collect { case c if !droppedSet.contains(c.makeColName()) => c.index }
          .zipWithIndex.toMap

        v.getColumnHistory().map { h =>
          val catGroupIndex = s.categoricalStats.zipWithIndex.collectFirst {
            case (groupStats, index) if groupStats.categoricalFeatures.contains(h.columnName) => index
          }
          val catIndexWithinGroup = catGroupIndex match {
            case Some(groupIndex) =>
              Some(s.categoricalStats(groupIndex).categoricalFeatures.indexOf(h.columnName))
            case _ => None
          }
          val keptIndex = indexInToIndexKept.get(h.index)

          h.parentFeatureOrigins ->
            Insights(
              derivedFeatureName = h.columnName,
              stagesApplied = h.parentFeatureStages,
              derivedFeatureGroup = h.indicatorGroup,
              derivedFeatureValue = h.indicatorValue,
              excluded = Option(s.dropped.contains(h.columnName)),
              corr = getCorr(s.correlationsWLabel, h.columnName),
              cramersV = catGroupIndex.map(i => s.categoricalStats(i).cramersV),
              mutualInformation = catGroupIndex.map(i => s.categoricalStats(i).mutualInfo),
              pointwiseMutualInformation = (catGroupIndex, catIndexWithinGroup) match {
                case (Some(groupIdx), Some(idx)) =>
                  getIfExists(idx, s.categoricalStats(groupIdx).pointwiseMutualInfo)
                case _ => Map.empty[String, Double]
              },
              countMatrix = (catGroupIndex, catIndexWithinGroup) match {
                case (Some(groupIdx), Some(idx)) =>
                  getIfExists(idx, s.categoricalStats(groupIdx).contingencyMatrix)
                case _ => Map.empty[String, Double]
              },
              contribution = keptIndex.map(i => contributions.map(_.applyOrElse(i, Seq.empty))).getOrElse(Seq.empty),
              min = getIfExists(h.index, s.featuresStatistics.min),
              max = getIfExists(h.index, s.featuresStatistics.max),
              mean = getIfExists(h.index, s.featuresStatistics.mean),
              variance = getIfExists(h.index, s.featuresStatistics.variance)
            )
        }
      case (Some(v), None) => v.getColumnHistory().map { h =>
        h.parentFeatureOrigins ->
          Insights(
            derivedFeatureName = h.columnName,
            stagesApplied = h.parentFeatureStages,
            derivedFeatureGroup = h.indicatorGroup,
            derivedFeatureValue = h.indicatorValue,
            contribution = contributions.map(_.applyOrElse(h.index, Seq.empty)) // nothing dropped without sanity check
          )
      }
      case (None, _) => Seq.empty
    }

    val blacklistInsights = blacklistedFeatures.map{ f =>
      Seq(f.name) -> Insights(derivedFeatureName = f.name, stagesApplied = Seq.empty, derivedFeatureGroup = None,
        derivedFeatureValue = None, excluded = Some(true))
    }

    val allInsights = featureInsights ++ blacklistInsights
    val allFeatures = rawFeatures ++ blacklistedFeatures

    allInsights
      .flatMap { case (feature, insights) => feature.map(_ -> insights) }
      .groupBy(_._1)
      .map {
        case (fname, seq) =>
          val ftype = allFeatures.find(_.name == fname)
            .getOrElse(throw new RuntimeException(s"No raw feature with name $fname found in raw features"))
            .typeName
          FeatureInsights(featureName = fname, featureType = ftype, derivedFeatures = seq.map(_._2))
      }.toSeq
  }

  private def getIfExists[T](index: Int, values: Seq[T]): Option[T] =
    if (index >= 0) Option(values(index)) else None

  private def getIfExists(index: Int, values: Map[String, Array[Double]]): Map[String, Double] =
    if (index >= 0) values.mapValues(_ (index)) else Map.empty

  private def getCorr(corr: Correlations, name: String): Option[Double] = {
    getIfExists(corr.featuresIn.indexOf(name), corr.values).orElse {
      val j = corr.nanCorrs.indexOf(name)
      if (j >= 0) Option(Double.NaN)
      else throw new RuntimeException(s"Column name $name does not exist in summary correlations")
    }
  }

  private[op] def getModelContributions(model: Option[SelectedModel]): Seq[Seq[Double]] = {
    model.flatMap {
      _.getSparkMlStage().map {
        case m: LogisticRegressionModel => m.coefficientMatrix.rowIter.toSeq.map(_.toArray.toSeq)
        case m: RandomForestClassificationModel => Seq(m.featureImportances.toArray.toSeq)
        case m: NaiveBayesModel => m.theta.rowIter.toSeq.map(_.toArray.toSeq)
        case m: DecisionTreeClassificationModel => Seq(m.featureImportances.toArray.toSeq)
        case m: LinearRegressionModel => Seq(m.coefficients.toArray.toSeq)
        case m: DecisionTreeRegressionModel => Seq(m.featureImportances.toArray.toSeq)
        case m: GradientBoostedTreesModel => Seq.empty[Seq[Double]]
        case m: RandomForestRegressionModel => Seq(m.featureImportances.toArray.toSeq)
        case _ => Seq.empty[Seq[Double]]
      }
    }.getOrElse(Seq.empty[Seq[Double]])
  }

  private def getModelInfo(model: Option[SelectedModel]): Map[String, Any] = {
    model.map(_.getMetadata().getSummaryMetadata().wrapped.underlyingMap)
      .getOrElse(Map.empty)
  }

  private def getStageInfo(stages: Array[OPStage]): Map[String, Any] = {
    def getParams(stage: PipelineStage): Map[String, Any] =
      stage.extractParamMap().toSeq
        .collect{
          case p if p.param.name != OpPipelineStageParamsNames.OutputMetadata &&
            p.param.name != OpPipelineStageParamsNames.InputSchema => p.param.name -> p.value
        }.toMap

    stages.map { s =>
      val params = s match {
        case m: Model[_] => getParams(if (m.hasParent) m.parent else m) // try for parent estimator so can get params
        case t: Transformer => getParams(t)
      }
      s.stageName -> Map("uid" -> s.uid, "params" -> params)
    }.toMap
  }
}
