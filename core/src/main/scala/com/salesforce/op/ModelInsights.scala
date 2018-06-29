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

import com.salesforce.op.evaluators._
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types.{OPVector, RealNN}
import com.salesforce.op.stages.impl.ModelsToTry
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry.{DecisionTree, LogisticRegression, NaiveBayes, RandomForest}
import com.salesforce.op.stages.impl.feature.TransmogrifierDefaults
import com.salesforce.op.stages.impl.preparators._
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry.{DecisionTreeRegression, GBTRegression, LinearRegression, RandomForestRegression}
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames._
import com.salesforce.op.stages.impl.selector.{ModelSelectorBase, SelectedModel}
import com.salesforce.op.stages.{OPStage, OpPipelineStageParams, OpPipelineStageParamsNames}
import com.salesforce.op.utils.json.JsonUtils
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.table.Table
import enumeratum._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.{Model, PipelineStage, Transformer}
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.sql.types.Metadata
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization.{write, writePretty}
import org.slf4j.LoggerFactory
import com.salesforce.op.utils.table.Alignment._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

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

  /**
   * Selected model UID
   */
  def selectedModelUID: String = selectedModelInfo(BestModelUid).toString

  /**
   * Selected model name
   */
  def selectedModelName: String = selectedModelInfo(BestModelName).toString

  /**
   * Selected model type, i.e. LogisticRegression, RandomForest etc.
   */
  def selectedModelType: ModelsToTry = modelType(selectedModelName).get

  /**
   * Selected model validation results computed during Cross Validation or Train Validation Split
   */
  def selectedModelValidationResults: Map[String, String] = validationResults(selectedModelName)

  /**
   * Train set evaluation metrics for selected model
   */
  def selectedModelTrainEvalMetrics: EvaluationMetrics = evaluationMetrics(TrainingEval)

  /**
   * Test set evaluation metrics (if any) for selected model
   */
  def selectedModelTestEvalMetrics: Option[EvaluationMetrics] = {
    selectedModelInfo.get(HoldOutEval).map(_ => evaluationMetrics(HoldOutEval))
  }

  /**
   * Validation results for all models computed during Cross Validation or Train Validation Split
   *
   * @return validation results keyed by model name
   */
  def validationResults: Map[String, Map[String, String]] = {
    val res = for {
      results <- getMap[String, Any](selectedModelInfo, TrainValSplitResults).recoverWith {
        case e => getMap[String, Any](selectedModelInfo, CrossValResults)
      }
    } yield results.keys.map(k => k -> getMap[String, String](results, k).getOrElse(Map.empty))
    res match {
      case Failure(e) => throw new Exception(s"Failed to extract validation results", e)
      case Success(ok) => ok.toMap
    }
  }

  /**
   * Validation results for a specified model type computed during Cross Validation or Train Validation Split
   *
   * @return validation results keyed by model name
   */
  def validationResults(mType: ModelsToTry): Map[String, Map[String, String]] = {
    validationResults.filter { case (modelName, _) => modelType(modelName).toOption.contains(mType) }
  }

  /**
   * All validated model types
   */
  def validatedModelTypes: Set[ModelsToTry] =
    validationResults.keys.flatMap(modelName => modelType(modelName).toOption).toSet

  /**
   * Validation type, i.e TrainValidationSplit, CrossValidation
   */
  def validationType: ValidationType = {
    if (getMap[String, Any](selectedModelInfo, TrainValSplitResults).isSuccess) ValidationType.TrainValidationSplit
    else if (getMap[String, Any](selectedModelInfo, CrossValResults).isSuccess) ValidationType.CrossValidation
    else throw new Exception(s"Failed to determine validation type")
  }

  /**
   * Evaluation metric type, i.e. AuPR, AuROC, F1 etc.
   */
  def evaluationMetricType: EnumEntry with EvalMetric = {
    val knownEvalMetrics = {
      (BinaryClassEvalMetrics.values ++ MultiClassEvalMetrics.values ++ RegressionEvalMetrics.values)
        .map(m => m.humanFriendlyName -> m).toMap
    }
    val evalMetrics = validationResults.flatMap(_._2.keys).flatMap(knownEvalMetrics.get).toSet.toList
    evalMetrics match {
      case evalMetric :: Nil => evalMetric
      case Nil => throw new Exception("Unable to determine evaluation metric type: no metrics were found")
      case metrics => throw new Exception(
        s"Unable to determine evaluation metric type since: multiple metrics were found - " + metrics.mkString(","))
    }
  }

  /**
   * Problem type, i.e. Binary Classification, Multi Classification or Regression
   */
  def problemType: ProblemType = selectedModelTrainEvalMetrics match {
    case _: BinaryClassificationMetrics => ProblemType.BinaryClassification
    case _: MultiClassificationMetrics => ProblemType.MultiClassification
    case _: RegressionMetrics => ProblemType.Regression
    case _ => ProblemType.Unknown
  }

  /**
   * Serialize to json string
   *
   * @param pretty should pretty format
   * @return json string
   */
  def toJson(pretty: Boolean = true): String = {
    implicit val formats = DefaultFormats
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
    res += prettySelectedModelInfo
    res += modelEvaluationMetrics
    res ++= topKCorrelations(topK)
    res ++= topKContributions(topK)
    res ++= topKCramersV(topK)
    res.mkString("\n")
  }

  private def prettyValidationResults: Seq[String] = {
    val evalSummary = {
      val vModelTypes = validatedModelTypes
      "Evaluated %s model%s using %s and %s metric.".format(
        vModelTypes.mkString(", "),
        if (vModelTypes.size > 1) "s" else "",
        validationType.humanFriendlyName, // TODO add number of folds or train/split ratio if possible
        evaluationMetricType.humanFriendlyName
      )
    }
    val modelEvalRes = for {
      modelType <- validatedModelTypes
      modelValidationResults = validationResults(modelType)
      evalMetric = evaluationMetricType.humanFriendlyName
    } yield {
      val evalMetricValues = modelValidationResults.flatMap { case (_, metrics) =>
        metrics.get(evalMetric).flatMap(v => Try(v.toDouble).toOption)
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

  private def prettySelectedModelInfo: String = {
    val bestModelType = selectedModelType
    val name = s"Selected Model - $bestModelType"
    val validationResults = selectedModelValidationResults.toSeq ++ Seq(
      "name" -> selectedModelName,
      "uid" -> selectedModelUID,
      "modelType" -> selectedModelType
    )
    val table = Table(name = name, columns = Seq("Model Param", "Value"), rows = validationResults.sortBy(_._1))
    table.prettyString()
  }

  private def modelEvaluationMetrics: String = {
    val name = "Model Evaluation Metrics"
    val trainEvalMetrics = selectedModelTrainEvalMetrics
    val testEvalMetrics = selectedModelTestEvalMetrics
    val (metricNameCol, holdOutCol, trainingCol) = ("Metric Name", "Hold Out Set Value", "Training Set Value")
    val trainMetrics = trainEvalMetrics.toMap.collect { case (k, v: Double) => k -> v.toString }.toSeq.sortBy(_._1)
    val table = testEvalMetrics match {
      case Some(testMetrics) =>
        val testMetricsMap = testMetrics.toMap
        val rows = trainMetrics.map { case (k, v) => (k, v, testMetricsMap(k).toString) }
        Table(name = name, columns = Seq(metricNameCol, trainingCol, holdOutCol), rows = rows)
      case None =>
        Table(name = name, columns = Seq(metricNameCol, trainingCol), rows = trainMetrics)
    }
    table.prettyString()
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

  private def modelType(modelName: String): Try[ModelsToTry] = Try {
    classificationModelType.orElse(regressionModelType).lift(modelName).getOrElse(
      throw new Exception(s"Unsupported model type for best model '$modelName'"))
  }

  private def classificationModelType: PartialFunction[String, ClassificationModelsToTry] = {
    case v if v.startsWith("logreg") => LogisticRegression
    case v if v.startsWith("rfc") => RandomForest
    case v if v.startsWith("dtc") => DecisionTree
    case v if v.startsWith("nb") => NaiveBayes
  }
  private def regressionModelType: PartialFunction[String, RegressionModelsToTry] = {
    case v if v.startsWith("linReg") => LinearRegression
    case v if v.startsWith("rfr") => RandomForestRegression
    case v if v.startsWith("dtr") => DecisionTreeRegression
    case v if v.startsWith("gbtr") => GBTRegression
  }
  private def evaluationMetrics(metricsName: String): EvaluationMetrics = {
    val res = for {
      metricsMap <- getMap[String, Any](selectedModelInfo, metricsName)
      evalMetrics <- Try(toEvaluationMetrics(metricsMap))
    } yield evalMetrics
    res match {
      case Failure(e) => throw new Exception(s"Failed to extract '$metricsName' metrics", e)
      case Success(ok) => ok
    }
  }
  private def getMap[K, V](m: Map[String, Any], name: String): Try[Map[K, V]] = Try {
    m(name) match {
      case m: Map[String, Any]@unchecked => m("map").asInstanceOf[Map[K, V]]
      case m: Metadata => m.underlyingMap.asInstanceOf[Map[K, V]]
    }
  }

  private val MetricName = "\\((.*)\\)\\_(.*)".r

  private def toEvaluationMetrics(metrics: Map[String, Any]): EvaluationMetrics = {
    import OpEvaluatorNames._
    val metricsType = metrics.keys.headOption match {
      case Some(MetricName(t, _)) if Set(binary, multi, regression).contains(t) => t
      case v => throw new Exception(s"Invalid model metric '$v'")
    }
    def parse[T <: EvaluationMetrics : ClassTag] = {
      val vals = metrics.map { case (MetricName(_, name), value) => name -> value }
      val valsJson = JsonUtils.toJsonString(vals)
      JsonUtils.fromString[T](valsJson).get
    }
    metricsType match {
      case `binary` => parse[BinaryClassificationMetrics]
      case `multi` => parse[MultiClassificationMetrics]
      case `regression` => parse[RegressionMetrics]
      case t => throw new Exception(s"Unsupported metrics type '$t'")
    }
  }
}

sealed trait ProblemType extends EnumEntry with Serializable
  object ProblemType extends Enum[ProblemType] {
  val values = findValues
  case object BinaryClassification extends ProblemType
  case object MultiClassification extends ProblemType
  case object Regression extends ProblemType
  case object Unknown extends ProblemType
}

sealed abstract class ValidationType(val humanFriendlyName: String) extends EnumEntry with Serializable
object ValidationType extends Enum[ValidationType] {
  val values = findValues
  case object CrossValidation extends ValidationType("Cross Validation")
  case object TrainValidationSplit extends ValidationType("Train Validation Split")
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
