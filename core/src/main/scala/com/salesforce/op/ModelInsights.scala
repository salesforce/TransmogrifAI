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

package com.salesforce.op

import com.salesforce.op.evaluators._
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.filters._
import com.salesforce.op.stages._
import com.salesforce.op.stages.impl.feature.TransmogrifierDefaults
import com.salesforce.op.stages.impl.preparators._
import com.salesforce.op.stages.impl.selector._
import com.salesforce.op.stages.impl.tuning.{DataBalancerSummary, DataCutterSummary, DataSplitterSummary}
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.stages.sparkwrappers.specific.OpPredictorWrapperModel
import com.salesforce.op.utils.json.{EnumEntrySerializer, SpecialDoubleSerializer}
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.table.Alignment._
import com.salesforce.op.utils.table.Table
import ml.dmlc.xgboost4j.scala.spark.OpXGBoost.RichBooster
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostRegressionModel}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.{Model, PipelineStage, Transformer}
import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization._
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
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
  selectedModelInfo: Option[ModelSelectorSummary],
  trainingParams: OpParams,
  stageInfo: Map[String, Any]
) {

  /**
   * Serialize to json string
   *
   * @param pretty should pretty format
   * @return json string
   */
  def toJson(pretty: Boolean = true): String = {
    implicit val formats = ModelInsights.SerializationFormats
    if (pretty) writePretty(this) else write(this)
  }

  /**
   * High level model summary in a compact print friendly format containing:
   * selected model info, model evaluation results and feature correlations/contributions/cramersV values.
   *
   * @param topK top K of feature correlations/contributions/cramersV values
   * @return high level model summary in a compact print friendly format
   */
  def prettyPrint(topK: Int = 15): String = {
    val res = new ArrayBuffer[String]()
    res ++= prettyValidationResults
    res ++= prettySelectedModelInfo
    res ++= modelEvaluationMetrics
    res ++= topKCorrelations(topK)
    res ++= topKContributions(topK)
    res ++= topKCramersV(topK)
    res.mkString("\n")
  }

  private def validatedModelTypes = selectedModelInfo.map(_.validationResults.map(_.modelType).toList.distinct)
    .getOrElse(List.empty)
  private def evaluationMetric = selectedModelInfo.map(_.evaluationMetric.humanFriendlyName)
  private def validationResults(modelType: String) = selectedModelInfo
    .map(_.validationResults.filter(_.modelType == modelType).toList).getOrElse(List.empty)

  private def prettyValidationResults: Seq[String] = {
    val evalSummary = {
      val vModelTypes = validatedModelTypes
      for {
        ev <- selectedModelInfo.map(_.validationType.humanFriendlyName)
        met <- evaluationMetric
      } yield {
        "Evaluated %s model%s using %s and %s metric.".format(
          vModelTypes.mkString(", "),
          if (vModelTypes.size > 1) "s" else "",
          ev, // TODO add number of folds or train/split ratio
          met
        )
      }
    }.getOrElse("No model selector found")
    val modelEvalRes = for {
      modelType <- validatedModelTypes
      modelValidationResults = validationResults(modelType)
      evalMetric <- evaluationMetric
    } yield {
      val evalMetricValues = modelValidationResults.map { eval =>
        eval.metricValues.asInstanceOf[SingleMetric].value
      }
      val minMetricValue = evalMetricValues.reduceOption[Double](math.min).getOrElse(Double.NaN)
      val maxMetricValue = evalMetricValues.reduceOption[Double](math.max).getOrElse(Double.NaN)

      "Evaluated %d %s model%s with %s metric between [%s, %s].".format(
        modelValidationResults.size,
        modelType,
        if (modelValidationResults.size > 1) "s" else "",
        evalMetric,
        minMetricValue,
        maxMetricValue
      )
    }
    Seq(evalSummary, modelEvalRes.mkString("\n"))
  }

  private def prettySelectedModelInfo: Seq[String] = {
    val excludedParams = Set(
      SparkWrapperParams.SparkStageParamName,
      ModelSelectorNames.outputParamName, ModelSelectorNames.inputParam1Name,
      ModelSelectorNames.inputParam2Name, ModelSelectorNames.outputParamName,
      OpPipelineStageParamsNames.InputFeatures, OpPipelineStageParamsNames.InputSchema,
      OpPipelineStageParamsNames.OutputMetadata,
      "labelCol", "predictionCol", "predictionValueCol", "rawPredictionCol", "probabilityCol"
    )
    val name = selectedModelInfo.map(sm => s"Selected Model - ${sm.bestModelType}").getOrElse("")
    val validationResults = (for {
      sm <- selectedModelInfo.toSeq
      e <- sm.validationResults.filter(v =>
        v.modelUID == sm.bestModelUID && v.modelName == sm.bestModelName && v.modelType == sm.bestModelType
      )
    } yield {
      val params = e.modelParameters.filterKeys(!excludedParams.contains(_))
      Seq("name" -> e.modelName, "uid" -> e.modelUID, "modelType" -> e.modelType) ++ params
    }).flatten.sortBy(_._1)
    if (validationResults.nonEmpty) {
      val table = Table(name = name, columns = Seq("Model Param", "Value"), rows = validationResults)
      Seq(table.prettyString())
    } else Seq.empty
  }

  private def modelEvaluationMetrics: Seq[String] = {
    val name = "Model Evaluation Metrics"
    val niceMetricsNames = {
      BinaryClassEvalMetrics.values ++ MultiClassEvalMetrics.values ++
        RegressionEvalMetrics.values ++ OpEvaluatorNames.values
    }.map(m => m.entryName -> m.humanFriendlyName).toMap
    def niceName(nm: String): String = nm.split('_').lastOption.flatMap(niceMetricsNames.get).getOrElse(nm)
    val trainEvalMetrics = selectedModelInfo.map(_.trainEvaluation)
    val testEvalMetrics = selectedModelInfo.flatMap(_.holdoutEvaluation)
    val (metricNameCol, holdOutCol, trainingCol) = ("Metric Name", "Hold Out Set Value", "Training Set Value")
    (trainEvalMetrics, testEvalMetrics) match {
      case (Some(trainMetrics), Some(testMetrics)) =>
        val trainMetricsMap = trainMetrics.toMap.collect { case (k, v: Double) => k -> v.toString }
        val testMetricsMap = testMetrics.toMap
        val rows = trainMetricsMap
          .map { case (k, v) => (niceName(k), v, testMetricsMap(k).toString) }.toSeq.sortBy(_._1)
        Seq(Table(name = name, columns = Seq(metricNameCol, trainingCol, holdOutCol), rows = rows).prettyString())
      case (Some(trainMetrics), None) =>
        val rows = trainMetrics.toMap.collect { case (k, v: Double) => niceName(k) -> v.toString }.toSeq.sortBy(_._1)
        Seq(Table(name = name, columns = Seq(metricNameCol, trainingCol), rows = rows).prettyString())
      case _ =>
        Seq.empty
    }
  }

  private def topKInsights(s: Seq[(FeatureInsights, Insights, Double)], topK: Int): Seq[(String, Double)] = {
    s.foldLeft(Seq.empty[(String, Double)]) {
      case (acc, (feature, derived, corr)) =>
        val insightValue = derived.derivedFeatureGroup -> derived.derivedFeatureValue match {
          case (Some(group), Some(OpVectorColumnMetadata.NullString)) => s"${feature.featureName}($group = null)"
          case (Some(group), Some(TransmogrifierDefaults.OtherString)) => s"${feature.featureName}($group = other)"
          case (Some(group), Some(value)) => s"${feature.featureName}($group = $value)"
          case (Some(group), None) => s"${feature.featureName}(group = $group)" // should not happen
          case (None, Some(value)) => s"${feature.featureName}(value = $value)" // should not happen
          case (None, None) => feature.featureName
        }
        if (acc.exists(_._1 == insightValue)) acc else acc :+ (insightValue, corr)
    } take topK
  }

  private def topKCorrelations(topK: Int): Seq[String] = {
    val corrs = for {
      (feature, derived) <- derivedNonExcludedFeatures
    } yield (feature, derived, derived.corr.collect { case v if !v.isNaN => v })

    val corrDsc = corrs.map { case (f, d, corr) => (f, d, corr.getOrElse(Double.MinValue)) }.sortBy(_._3).reverse
    val corrAsc = corrs.map { case (f, d, corr) => (f, d, corr.getOrElse(Double.MaxValue)) }.sortBy(_._3)
    val topPositiveCorrs = topKInsights(corrDsc, topK)
    val topNegativeCorrs = topKInsights(corrAsc, topK).filterNot(topPositiveCorrs.contains)

    val correlationCol = "Correlation Value"

    lazy val topPositive = Table(
      name = "Top Model Insights",
      columns = Seq("Top Positive Correlations", correlationCol),
      rows = topPositiveCorrs
    ).prettyString(columnAlignments = Map(correlationCol -> Right))

    lazy val topNegative = Table(
      columns = Seq("Top Negative Correlations", correlationCol),
      rows = topNegativeCorrs
    ).prettyString(columnAlignments = Map(correlationCol -> Right))

    if (topNegativeCorrs.isEmpty) Seq(topPositive) else Seq(topPositive, topNegative)
  }

  private def topKContributions(topK: Int): Option[String] = {
    val contribs = for {
      (feature, derived) <- derivedNonExcludedFeatures
      contrib = math.abs(derived.contribution.reduceOption[Double](math.max).getOrElse(0.0))
    } yield (feature, derived, contrib)

    val contribDesc = contribs.sortBy(_._3).reverse
    val rows = topKInsights(contribDesc, topK)
    numericalTable(columns = Seq("Top Contributions", "Contribution Value"), rows)
  }

  private def topKCramersV(topK: Int): Option[String] = {
    val cramersV = for {
      (feature, derived) <- derivedNonExcludedFeatures
      group <- derived.derivedFeatureGroup
      cramersV <- derived.cramersV
    } yield group -> cramersV

    val topCramersV = cramersV.distinct.sortBy(_._2).reverse.take(topK)
    numericalTable(columns = Seq("Top CramersV", "CramersV"), rows = topCramersV)
  }

  private def derivedNonExcludedFeatures: Seq[(FeatureInsights, Insights)] = {
    for {
      feature <- features
      derived <- feature.derivedFeatures
      if !derived.excluded.contains(true)
    } yield feature -> derived
  }

  private def numericalTable(columns: Seq[String], rows: Seq[(String, Double)]): Option[String] =
    if (rows.isEmpty) None else Some(Table(columns, rows).prettyString(columnAlignments = Map(columns.last -> Right)))

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
 * @param featureName      name of raw feature insights are about
 * @param featureType      type of raw feature insights are about
 * @param derivedFeatures  sequence containing insights for each feature derived from the raw feature
 * @param metrics          sequence containing metrics computed in RawFeatureFilter
 * @param distributions    distribution information for the raw feature (if calculated in RawFeatureFilter)
 * @param exclusionReasons exclusion reasons for the raw feature (if calculated in RawFeatureFilter)
 *
 */
case class FeatureInsights
(
  featureName: String,
  featureType: String,
  derivedFeatures: Seq[Insights],
  metrics: Seq[RawFeatureFilterMetrics] = Seq.empty,
  distributions: Seq[FeatureDistribution] = Seq.empty,
  exclusionReasons: Seq[ExclusionReasons] = Seq.empty
)

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

  val SerializationFormats: Formats = {
    val typeHints = FullTypeHints(List(
      classOf[Continuous], classOf[Discrete],
      classOf[DataBalancerSummary], classOf[DataCutterSummary], classOf[DataSplitterSummary],
      classOf[SingleMetric], classOf[MultiMetrics], classOf[BinaryClassificationMetrics],
      classOf[BinaryClassificationBinMetrics], classOf[ThresholdMetrics],
      classOf[MultiClassificationMetrics], classOf[RegressionMetrics]
    ))
    val evalMetricsSerializer = new CustomSerializer[EvalMetric](_ =>
      ( { case JString(s) => EvalMetric.withNameInsensitive(s) },
        { case x: EvalMetric => JString(x.entryName) }
      )
    )
    Serialization.formats(typeHints) +
      EnumEntrySerializer.json4s[ValidationType](ValidationType) +
      EnumEntrySerializer.json4s[ProblemType](ProblemType) +
      new SpecialDoubleSerializer +
      evalMetricsSerializer
  }

  /**
   * Read ModelInsights from a json
   *
   * @param json model insights in json
   * @return Try[ModelInsights]
   */
  def fromJson(json: String): Try[ModelInsights] = {
    implicit val formats: Formats = SerializationFormats
    Try { read[ModelInsights](json) }
  }

  /**
   * Function to extract the model summary info from the stages used to create the selected model output feature
   *
   * @param stages                  stages used to make the feature
   * @param rawFeatures             raw features in the workflow
   * @param trainingParams          parameters used to create the workflow model
   * @param blacklistedFeatures     blacklisted features from use in DAG
   * @param blacklistedMapKeys      blacklisted map keys from use in DAG
   * @param rawFeatureFilterResults results of raw feature filter
   * @return
   */
  private[op] def extractFromStages(
    stages: Array[OPStage],
    rawFeatures: Array[features.OPFeature],
    trainingParams: OpParams,
    blacklistedFeatures: Array[features.OPFeature],
    blacklistedMapKeys: Map[String, Set[String]],
    rawFeatureFilterResults: RawFeatureFilterResults
  ): ModelInsights = {
    val sanityCheckers = stages.collect { case s: SanityCheckerModel => s }
    val sanityChecker = sanityCheckers.lastOption
    val checkerSummary = sanityChecker.map(s => SanityCheckerSummary.fromMetadata(s.getMetadata().getSummaryMetadata()))
    log.info(
      s"Found ${sanityCheckers.length} sanity checkers will " +
        s"${sanityChecker.map("use results from the last checker:" + _.uid + "to").getOrElse("not")}" +
        s" to fill in model insights"
    )

    val models: Array[OPStage with Model[_]] = stages.collect{
      case s: SelectedModel => s
      case s: OpPredictorWrapperModel[_] => s
    } // TODO support other model types?
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
        .orElse(model.flatMap(m => makeMeta(m.parent.asInstanceOf[ModelSelector[_, _]])))
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

    val labelSummary = getLabelSummary(label, checkerSummary)

    ModelInsights(
      label = labelSummary,
      features = getFeatureInsights(vectorInput, checkerSummary, model, rawFeatures,
        blacklistedFeatures, blacklistedMapKeys, rawFeatureFilterResults, labelSummary),
      selectedModelInfo = getModelInfo(model),
      trainingParams = trainingParams,
      stageInfo = RawFeatureFilterConfig.toStageInfo(rawFeatureFilterResults.rawFeatureFilterConfig)
        ++ getStageInfo(stages)
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
    model: Option[Model[_]],
    rawFeatures: Array[features.OPFeature],
    blacklistedFeatures: Array[features.OPFeature],
    blacklistedMapKeys: Map[String, Set[String]],
    rawFeatureFilterResults: RawFeatureFilterResults = RawFeatureFilterResults(),
    label: LabelSummary
  ): Seq[FeatureInsights] = {
    val featureInsights = (vectorInfo, summary) match {
      case (Some(v), Some(s)) =>
        val contributions = getModelContributions(model, Option(v.columns.length))
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
          val featureStd = math.sqrt(getIfExists(h.index, s.featuresStatistics.variance).getOrElse(1.0))
          val sparkFtrContrib = keptIndex
            .map(i => contributions.map(_.applyOrElse(i, (_: Int) => 0.0))).getOrElse(Seq.empty)
          val defaultLabelStd = 1.0
          val labelStd = label.distribution match {
            case Some(Continuous(_, _, _, variance)) =>
              if (variance == 0) {
                log.warn("The standard deviation of the label is zero, " +
                  "so the coefficients and intercepts of the model will be zeros, training is not needed.")
                defaultLabelStd
              }
              else math.sqrt(variance)
            case Some(Discrete(domain, prob)) =>
              // mean = sum (x_i * p_i)
              val mean = (domain zip prob).foldLeft(0.0) {
                case (weightSum, (d, p)) => weightSum + d.toDouble * p
              }
              // variance = sum (x_i - mu)^2 * p_i
              val discreteVariance = (domain zip prob).foldLeft(0.0) {
                case (sqweightSum, (d, p)) => sqweightSum + (d.toDouble - mean) * (d.toDouble - mean) * p
              }
              if (discreteVariance == 0) {
                log.warn("The standard deviation of the label is zero, " +
                  "so the coefficients and intercepts of the model will be zeros, training is not needed.")
                defaultLabelStd
              }
              else math.sqrt(discreteVariance)
            case Some(_) => {
              log.warn("Failing to perform weight descaling because distribution is unsupported.")
              defaultLabelStd
            }
            case None => {
              log.warn("Label does not exist, please check your data")
              defaultLabelStd
            }
          }

          h.parentFeatureOrigins ->
            Insights(
              derivedFeatureName = h.columnName,
              stagesApplied = h.parentFeatureStages,
              derivedFeatureGroup = h.grouping,
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
              contribution =
                descaleLRContrib(model, sparkFtrContrib, featureStd, labelStd).getOrElse(sparkFtrContrib),

              min = getIfExists(h.index, s.featuresStatistics.min),
              max = getIfExists(h.index, s.featuresStatistics.max),
              mean = getIfExists(h.index, s.featuresStatistics.mean),
              variance = getIfExists(h.index, s.featuresStatistics.variance)
            )
        }
      case (Some(v), None) =>
        val contributions = getModelContributions(model, Option(v.columns.length))
        v.getColumnHistory().map { h =>
          h.parentFeatureOrigins -> Insights(
            derivedFeatureName = h.columnName,
            stagesApplied = h.parentFeatureStages,
            derivedFeatureGroup = h.grouping,
            derivedFeatureValue = h.indicatorValue,
            contribution =
              contributions.map(_.applyOrElse(h.index, (_: Int) => 0.0)) // nothing dropped without sanity check
          )
      }
      case (None, _) => Seq.empty
    }

    val blacklistInsights = blacklistedFeatures.map{ f =>
      Seq(f.name) -> Insights(derivedFeatureName = f.name, stagesApplied = Seq.empty, derivedFeatureGroup = None,
        derivedFeatureValue = None, excluded = Some(true))
    }

    val blacklistMapInsights = blacklistedMapKeys.toArray.flatMap { case (mname, keys) =>
      keys.toArray.map(key => {
        Seq(mname) ->
          Insights(derivedFeatureName = key, stagesApplied = Seq.empty, derivedFeatureGroup = Some(key),
            derivedFeatureValue = None, excluded = Some(true))
      })
    }

    val allInsights = featureInsights ++ blacklistInsights ++ blacklistMapInsights
    val allFeatures = rawFeatures ++ blacklistedFeatures

    allInsights
      .flatMap { case (feature, insights) => feature.map(_ -> insights) }
      .groupBy(_._1)
      .map {
        case (fname, seq) =>
          val ftype = allFeatures.find(_.name == fname)
            .map(_.typeName)
            .getOrElse("")
          val metrics = rawFeatureFilterResults.rawFeatureFilterMetrics.filter(_.name == fname)
          val distributions = rawFeatureFilterResults.rawFeatureDistributions.filter(_.name == fname)
          val exclusionReasons = rawFeatureFilterResults.exclusionReasons.filter(_.name == fname)
          FeatureInsights(featureName = fname, featureType = ftype, derivedFeatures = seq.map(_._2),
            metrics = metrics, distributions = distributions, exclusionReasons = exclusionReasons)
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
      else None
    }
  }

  private[op] def descaleLRContrib(
    model: Option[Model[_]],
    sparkFtrContrib: Seq[Double],
    featureStd: Double,
    labelStd: Double): Option[Seq[Double]] = {
    val stage = model.flatMap {
      case m: SparkWrapperParams[_] => m.getSparkMlStage()
      case _ => None
    }
    stage.collect {
      case m: LogisticRegressionModel =>
        if (m.getStandardization && sparkFtrContrib.nonEmpty) {
          // scale entire feature contribution vector
          // See https://think-lab.github.io/d/205/
          // § 4.5.2 Standardized Interpretations, An Introduction to Categorical Data Analysis, Alan Agresti
          sparkFtrContrib.map(_ * featureStd)
        }
        else sparkFtrContrib
      case m: LinearRegressionModel =>
        if (m.getStandardization && sparkFtrContrib.nonEmpty) {
          // need to also divide by labelStd for linear regression
          // See https://u.demog.berkeley.edu/~andrew/teaching/standard_coeff.pdf
          // See https://en.wikipedia.org/wiki/Standardized_coefficient
        sparkFtrContrib.map(_ * featureStd / labelStd)
      }
      else sparkFtrContrib
      case _ => sparkFtrContrib
    }
  }

  private[op] def getModelContributions
  (model: Option[Model[_]], featureVectorSize: Option[Int] = None): Seq[Seq[Double]] = {
    val stage = model.flatMap {
      case m: SparkWrapperParams[_] => m.getSparkMlStage()
      case _ => None
    }
    val contributions = stage.collect {
      case m: LogisticRegressionModel => m.coefficientMatrix.rowIter.toSeq.map(_.toArray.toSeq)
      case m: RandomForestClassificationModel => Seq(m.featureImportances.toArray.toSeq)
      case m: NaiveBayesModel => m.theta.rowIter.toSeq.map(_.toArray.toSeq)
      case m: DecisionTreeClassificationModel => Seq(m.featureImportances.toArray.toSeq)
      case m: GBTClassificationModel => Seq(m.featureImportances.toArray.toSeq)
      case m: LinearSVCModel => Seq(m.coefficients.toArray.toSeq)
      case m: LinearRegressionModel => Seq(m.coefficients.toArray.toSeq)
      case m: DecisionTreeRegressionModel => Seq(m.featureImportances.toArray.toSeq)
      case m: RandomForestRegressionModel => Seq(m.featureImportances.toArray.toSeq)
      case m: GBTRegressionModel => Seq(m.featureImportances.toArray.toSeq)
      case m: GeneralizedLinearRegressionModel => Seq(m.coefficients.toArray.toSeq)
      case m: XGBoostRegressionModel => Seq(m.nativeBooster.getFeatureScoreVector(featureVectorSize).toArray.toSeq)
      case m: XGBoostClassificationModel => Seq(m.nativeBooster.getFeatureScoreVector(featureVectorSize).toArray.toSeq)
    }
    contributions.getOrElse(Seq.empty)
  }

  private def getModelInfo(model: Option[Model[_]]): Option[ModelSelectorSummary] = {
    model match {
      case Some(m: SelectedModel) =>
        Try(ModelSelectorSummary.fromMetadata(m.getMetadata().getSummaryMetadata())).toOption
      case _ => None
    }
  }

  private def getStageInfo(stages: Array[OPStage]): Map[String, Any] = {
    def getParams(stage: PipelineStage): Map[String, String] = {
      stage.extractParamMap().toSeq.collect {
        case p if p.param.name == OpPipelineStageParamsNames.InputFeatures =>
          p.param.name -> p.value.asInstanceOf[Array[TransientFeature]].map(_.toJsonString()).mkString(", ")
        case p if p.param.name != OpPipelineStageParamsNames.OutputMetadata &&
          p.param.name != OpPipelineStageParamsNames.InputSchema && Option(p.value).nonEmpty =>
          p.param.name -> p.value.toString
      }.toMap
    }
    stages.map { s =>
      val params = s match {
        case m: Model[_] => getParams(if (m.hasParent) m.parent else m) // try for parent estimator so can get params
        case t: Transformer => getParams(t)
      }
      s.stageName -> Map("uid" -> s.uid, "params" -> params)
    }.toMap
  }
}
