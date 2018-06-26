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

import com.salesforce.op.evaluators.{EvaluationMetrics, OpEvaluatorBase}
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.features.{FeatureLike, OPFeature}
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.stages.impl.feature.TransmogrifierDefaults
import com.salesforce.op.stages.{OPStage, OpPipelineStage, OpTransformer}
import com.salesforce.op.utils.spark.OpVectorColumnMetadata
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.stages.FitStagesUtil
import com.salesforce.op.utils.table.Alignment._
import com.salesforce.op.utils.table._
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s.JValue
import org.json4s.JsonAST.{JField, JObject}
import org.json4s.jackson.JsonMethods.{pretty, render}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Try


/**
 * Workflow model is a container and executor for the sequence of transformations that have been fit to the data
 * to produce the desired output features
 *
 * @param uid unique identifier for this workflow model
 * @param trainingParams params that were used during model training
 */
class OpWorkflowModel(val uid: String = UID[OpWorkflowModel], val trainingParams: OpParams) extends OpWorkflowCore {

  /**
   * Set reader parameters from OpWorkflowParams object for run (stage parameters passed in will have no effect)
   *
   * @param newParams new parameter values
   */
  final def setParameters(newParams: OpParams): this.type = {
    parameters = newParams
    log.info("Note that stage params have no effect on workflow models once trained: {}", parameters.stageParams)
    this
  }

  protected[op] def setFeatures(features: Array[OPFeature]): this.type = {
    resultFeatures = features
    rawFeatures = features.flatMap(_.rawFeatures).distinct.sortBy(_.name)
    this
  }

  protected[op] def setBlacklist(features: Array[OPFeature]): this.type = {
    blacklistedFeatures = features
    this
  }

  /**
   * Used to generate dataframe from reader and raw features list
   *
   * @return Dataframe with all the features generated + persisted
   */
  protected def generateRawData()(implicit spark: SparkSession): DataFrame = {
    require(reader.nonEmpty, "Data reader must be set")
    checkReadersAndFeatures()
    reader.get.generateDataFrame(rawFeatures, parameters).persist() // don't want to redo this
  }

  /**
   * Returns a dataframe containing all the columns generated up to and including the feature input
   *
   * @param feature input feature to compute up to
   * @throws IllegalArgumentException if a feature is not part of this workflow
   * @return Dataframe containing columns corresponding to all of the features generated up to the feature given
   */
  def computeDataUpTo(feature: OPFeature, persistEveryKStages: Int = OpWorkflowModel.PersistEveryKStages)
    (implicit spark: SparkSession): DataFrame = {
    if (findOriginStageId(feature).isEmpty) {
      log.warn("Could not find origin stage for feature in workflow!! Defaulting to generate raw features.")
      generateRawData()
    } else {
      val fittedFeature = feature.copyWithNewStages(stages)
      val dag = FitStagesUtil.computeDAG(Array(fittedFeature))
      applyTransformationsDAG(generateRawData(), dag, persistEveryKStages)
    }
  }

  /**
   * Gets the fitted stage that generates the input feature
   *
   * @param feature feature want the origin stage for
   * @tparam T Type of feature
   * @throws IllegalArgumentException if a feature is not part of this workflow
   * @return Fitted origin stage for feature
   */
  def getOriginStageOf[T <: FeatureType](feature: FeatureLike[T]): OpPipelineStage[T] = {
    findOriginStage(feature).asInstanceOf[OpPipelineStage[T]]
  }

  /**
   * Gets the updated version of a feature when the DAG has been modified with a raw feature filter
   *
   * @param features feature want a the updated history for
   * @throws IllegalArgumentException if a feature is not part of this workflow
   * @return Updated instance of feature
   */
  def getUpdatedFeatures(features: Array[OPFeature]): Array[OPFeature] = {
    val allFeatures = rawFeatures ++ blacklistedFeatures ++ stages.map(_.getOutput())
    features.map{f => allFeatures.find(_.sameOrigin(f))
      .getOrElse(throw new IllegalArgumentException(s"feature $f is not a part of this workflow"))
    }
  }

  /**
   * Get the metadata associated with the features
   *
   * @param features features to get metadata for
   * @throws IllegalArgumentException if a feature is not part of this workflow
   * @return metadata associated with the features
   */
  def getMetadata(features: OPFeature*): Map[OPFeature, Metadata] = {
    features.map(feature => feature -> findOriginStage(feature).getMetadata()).toMap
  }

  /**
   * Get model insights for the model used to create the input feature.
   * Will traverse the DAG to find the LAST model selector and sanity checker used in
   * the creation of the selected feature
   *
   * @param feature feature to find model info for
   * @return Model insights class containing summary of modeling and sanity checking
   */
  def modelInsights(feature: OPFeature): ModelInsights = {
    findOriginStageId(feature) match {
      case None =>
        throw new IllegalArgumentException(
          s"Cannot produce model insights: feature '${feature.name}' (${feature.uid}) " +
            "is either a raw feature or not part of this workflow model"
        )
      case Some(_) =>
        val parentStageIds = feature.traverse[Set[String]](Set.empty[String])((s, f) => s + f.originStage.uid)
        val modelStages = stages.filter(s => parentStageIds.contains(s.uid))
        ModelInsights.extractFromStages(modelStages, rawFeatures, trainingParams, blacklistedFeatures)
    }
  }

  /**
   * Pulls all summary metadata of transformers and puts them in json
   *
   * @return json summary
   */
  def summaryJson(): JValue = JObject(
    stages.map(s => s.uid -> s.getMetadata()).collect {
      case (id, meta) if meta.containsSummaryMetadata() =>
        JField(id, meta.getSummaryMetadata().wrapped.jsonValue)
    }: _*
  )

  /**
   * Pulls all summary metadata of transformers and puts them into json string
   *
   * @return json string summary
   */
  def summary(): String = pretty(render(summaryJson()))

  /**
   * High level model summary in a compact print friendly format containing:
   * selected model info, model evaluation results and feature correlations/contributions/cramersV values.
   *
   * @param topK top K of feature correlations/contributions/cramersV values
   * @return high level model summary in a compact print friendly format
   */
  def summaryPretty(topK: Int = 15): String = {
    val response = resultFeatures.find(_.isResponse).getOrElse(throw new Exception("No response feature is defined"))
    val insights = modelInsights(response)
    val summary = new ArrayBuffer[String]()
    summary ++= validationResults(insights)
    summary += selectedModelInfo(insights)
    summary += modelEvaluationMetrics(insights)
    summary ++= topKCorrelations(insights, topK)
    summary ++= topKContributions(insights, topK)
    summary ++= topKCramersV(insights, topK)
    summary.mkString("\n")
  }

  private def validationResults(insights: ModelInsights): Seq[String] = {
    val evalSummary = {
      val validatedModelTypes = insights.validatedModelTypes
      val validationType = insights.validationType.humanFriendlyName
      val evalMetric = insights.evaluationMetricType.humanFriendlyName
      "Evaluated %s model%s using %s and %s metric.".format(
        validatedModelTypes.mkString(", "),
        if (validatedModelTypes.size > 1) "s" else "",
        validationType, // TODO add number of folds or train/split ratio if possible
        evalMetric
      )
    }
    val modelEvalRes = for {
      modelType <- insights.validatedModelTypes
      modelValidationResults = insights.validationResults(modelType)
      evalMetric = insights.evaluationMetricType.humanFriendlyName
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

  private def selectedModelInfo(insights: ModelInsights): String = {
    val bestModelType = insights.selectedModelType
    val name = s"Selected Model - $bestModelType"
    val validationResults = insights.selectedModelValidationResults.toSeq ++ Seq(
      "name" -> insights.selectedModelName,
      "uid" -> insights.selectedModelUID,
      "modelType" -> insights.selectedModelType
    )
    val table = Table(name = name, columns = Seq("Model Param", "Value"), rows = validationResults.sortBy(_._1))
    table.prettyString()
  }

  private def modelEvaluationMetrics(insights: ModelInsights): String = {
    val name = "Model Evaluation Metrics"
    val trainEvalMetrics = insights.selectedModelTrainEvalMetrics
    val testEvalMetrics = insights.selectedModelTestEvalMetrics
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

  private def topKCorrelations(insights: ModelInsights, topK: Int): Seq[String] = {
    val maxCorrs = insights.features
      .flatMap(f => f.derivedFeatures.map(d => (f, d, d.corr.getOrElse(Double.MinValue)))).sortBy(-_._3)
    val minCorrs = insights.features
      .flatMap(f => f.derivedFeatures.map(d => (f, d, d.corr.getOrElse(Double.MaxValue)))).sortBy(_._3)
    val topPositiveInsights = topKInsights(maxCorrs, topK)
    val topNegativeInsights = topKInsights(minCorrs, topK).filterNot(topPositiveInsights.contains)

    val correlationCol = "Correlation Value"

    lazy val topPositive = Table(
      name = "Top Model Insights",
      columns = Seq("Top Positive Correlations", correlationCol),
      rows = topPositiveInsights
    ).prettyString(columnAlignments = Map(correlationCol -> Right))

    lazy val topNegative = Table(
      columns = Seq("Top Negative Correlations", correlationCol),
      rows = topNegativeInsights
    ).prettyString(columnAlignments = Map(correlationCol -> Right))

    if (topNegativeInsights.isEmpty) Seq(topPositive) else Seq(topPositive, topNegative)
  }

  private def topKContributions(insights: ModelInsights, topK: Int): Option[String] = {
    val maxContribFeatures = insights.features
      .flatMap(f => f.derivedFeatures.map(d =>
        (f, d, d.contribution.reduceOption[Double](math.max).getOrElse(Double.MinValue))))
      .sortBy(v => -1 * math.abs(v._3))
    val rows = topKInsights(maxContribFeatures, topK)
    numericalTable(columns = Seq("Top Contributions", "Contribution Value"), rows)
  }

  private def topKCramersV(insights: ModelInsights, topK: Int): Option[String] = {
    val allCramersV = for {
      feature <- insights.features
      derived <- feature.derivedFeatures
      group <- derived.derivedFeatureGroup
      cramersV <- derived.cramersV
    } yield group -> cramersV
    val rows = allCramersV.sortBy(-_._2).take(topK)
    numericalTable(columns = Seq("Top CramersV", "CramersV"), rows)
  }

  private def numericalTable(columns: Seq[String], rows: Seq[(String, Double)]): Option[String] =
    if (rows.isEmpty) None else Some(Table(columns, rows).prettyString(columnAlignments = Map(columns.last -> Right)))

  /**
   * Save this model to a path
   *
   * @param path      path to save the model
   * @param overwrite should overwrite if the path exists
   */
  def save(path: String, overwrite: Boolean = true): Unit =
    OpWorkflowModelWriter.save(this, path = path, overwrite = overwrite)

  /**
   * Gets the fitted stage that generates the input feature
   *
   * @param feature feature want the origin stage for
   * @throws IllegalArgumentException if a feature is not part of this workflow
   * @return Fitted origin stage for feature
   */
  private def findOriginStage(feature: OPFeature): OPStage = {
    val stageId = findOriginStageId(feature)
    require(stageId.nonEmpty, s"Feature '${feature.name}' is not actually part of this workflow")
    stages(stageId.get)
  }

  /**
   * Load up the data as specified by the data reader then transform that data using the transformers specified in
   * this workflow. We will always keep the key and result features in the returned dataframe, but there are options
   * to keep the other raw & intermediate features.
   *
   * This method optimizes scoring by grouping applying bulks of [[OpTransformer]] stages on each step.
   * The rest of the stages go are applied sequentially (as [[org.apache.spark.ml.Pipeline]] does)
   *
   * @param path                     optional path to write out the scores to a file
   * @param keepRawFeatures          flag to enable keeping raw features in the output DataFrame as well
   * @param keepIntermediateFeatures flag to enable keeping intermediate features in the output DataFrame as well
   * @param persistEveryKStages      how often to break up catalyst by persisting the data
   *                                 (applies for non [[OpTransformer]] stages only),
   *                                 to turn off set to Int.MaxValue (not recommended)
   * @param persistScores            should persist the final scores dataframe
   * @return Dataframe that contains all the columns generated by the transformers in this workflow model as well as
   *         the key and result features, along with other features if the above flags are set to true.
   *
   */
  def score(
    path: Option[String] = None,
    keepRawFeatures: Boolean = OpWorkflowModel.KeepRawFeatures,
    keepIntermediateFeatures: Boolean = OpWorkflowModel.KeepIntermediateFeatures,
    persistEveryKStages: Int = OpWorkflowModel.PersistEveryKStages,
    persistScores: Boolean = OpWorkflowModel.PersistScores
  )(implicit spark: SparkSession): DataFrame = {
    val (scores, _) = scoreFn(
      keepRawFeatures = keepRawFeatures,
      keepIntermediateFeatures = keepIntermediateFeatures,
      persistEveryKStages = persistEveryKStages,
      persistScores = persistScores
    )(spark)(path)
    scores
  }

  /**
   * Load up the data as specified by the data reader then transform that data using the transformers specified in
   * this workflow. We will always keep the key and result features in the returned dataframe, but there are options
   * to keep the other raw & intermediate features.
   *
   * This method optimizes scoring by grouping applying bulks of [[OpTransformer]] stages on each step.
   * The rest of the stages go are applied sequentially (as [[org.apache.spark.ml.Pipeline]] does)
   *
   * @param evaluator                evalutator to use for metrics generation
   * @param path                     optional path to write out the scores to a file
   * @param keepRawFeatures          flag to enable keeping raw features in the output DataFrame as well
   * @param keepIntermediateFeatures flag to enable keeping intermediate features in the output DataFrame as well
   * @param persistEveryKStages      how often to break up catalyst by persisting the data
   *                                 (applies for non [[OpTransformer]] stages only),
   *                                 to turn off set to Int.MaxValue (not recommended)
   * @param persistScores            should persist the final scores dataframe
   * @param metricsPath              optional path to write out the metrics to a file
   * @return Dataframe that contains all the columns generated by the transformers in this workflow model as well as
   *         the key and result features, along with other features if the above flags are set to true.
   *         Also returns metrics computed with evaluator.
   */
  def scoreAndEvaluate(
    evaluator: OpEvaluatorBase[_ <: EvaluationMetrics],
    path: Option[String] = None,
    keepRawFeatures: Boolean = OpWorkflowModel.KeepRawFeatures,
    keepIntermediateFeatures: Boolean = OpWorkflowModel.KeepIntermediateFeatures,
    persistEveryKStages: Int = OpWorkflowModel.PersistEveryKStages,
    persistScores: Boolean = OpWorkflowModel.PersistScores,
    metricsPath: Option[String] = None
  )(implicit spark: SparkSession): (DataFrame, EvaluationMetrics) = {
    val (scores, metrics) = scoreFn(
      keepRawFeatures = keepRawFeatures,
      keepIntermediateFeatures = keepIntermediateFeatures,
      persistEveryKStages = persistEveryKStages,
      persistScores = persistScores,
      evaluator = Option(evaluator),
      metricsPath = metricsPath
    )(spark)(path)
    scores -> metrics.get
  }

  /**
   * Load up the data by the reader, transform it and then evaluate
   *
   * @param evaluator   OP Evaluator
   * @param metricsPath path to write out the metrics
   * @param spark       spark session
   * @return evaluation metrics
   */
  def evaluate[T <: EvaluationMetrics : ClassTag](
    evaluator: OpEvaluatorBase[T], metricsPath: Option[String] = None, scoresPath: Option[String] = None
  )(implicit spark: SparkSession): T = {
    val (_, eval) = scoreAndEvaluate(evaluator = evaluator, metricsPath = metricsPath, path = scoresPath)
    eval.asInstanceOf[T]
  }

  private[op] def scoreFn(
    keepRawFeatures: Boolean = OpWorkflowModel.KeepRawFeatures,
    keepIntermediateFeatures: Boolean = OpWorkflowModel.KeepIntermediateFeatures,
    persistEveryKStages: Int = OpWorkflowModel.PersistEveryKStages,
    persistScores: Boolean = OpWorkflowModel.PersistScores,
    evaluator: Option[OpEvaluatorBase[_ <: EvaluationMetrics]] = None,
    metricsPath: Option[String] = None
  )(implicit spark: SparkSession): Option[String] => (DataFrame, Option[EvaluationMetrics]) = {
    require(persistEveryKStages >= 1, s"persistEveryKStages value of $persistEveryKStages is invalid must be >= 1")

    // TODO: replace 'stages' with 'stagesDag'. (is a breaking change for serialization, but would simplify scoreFn)
    // Pre-compute transformations dag
    val dag = FitStagesUtil.computeDAG(resultFeatures)

    (path: Option[String]) => {
      // Generate the dataframe with raw features
      val rawData: DataFrame = generateRawData()

      // Apply the transformations DAG on raw data
      val transformedData: DataFrame = applyTransformationsDAG(rawData, dag, persistEveryKStages)

      // Save the scores
      val (scores, metrics) = saveScores(
        path = path,
        keepRawFeatures = keepRawFeatures,
        keepIntermediateFeatures = keepIntermediateFeatures,
        transformedData = transformedData,
        persistScores = persistScores,
        evaluator = evaluator,
        metricsPath = metricsPath
      )
      // Unpersist raw data, since it's not needed anymore
      rawData.unpersist()
      scores -> metrics
    }
  }


  /**
   * Function to remove unwanted columns from scored dataframe, evaluate and save results
   *
   * @param path                     optional path to write out the scores to a file
   * @param keepRawFeatures          flag to enable keeping raw features in the output DataFrame as well
   * @param keepIntermediateFeatures flag to enable keeping intermediate features in the output DataFrame as well
   * @param transformedData          transformed & scored dataframe
   * @param persistScores            should persist the final scores dataframe
   * @param evaluator                optional evaluator
   * @param metricsPath              optional path to write out the metrics to a file
   * @return cleaned up score dataframe & metrics
   */
  private def saveScores
  (
    path: Option[String],
    keepRawFeatures: Boolean,
    keepIntermediateFeatures: Boolean,
    transformedData: DataFrame,
    persistScores: Boolean,
    evaluator: Option[OpEvaluatorBase[_ <: EvaluationMetrics]],
    metricsPath: Option[String]
  )(implicit spark: SparkSession): (DataFrame, Option[EvaluationMetrics]) = {

    // Evaluate and save the metrics
    val metrics = for {
      ev <- evaluator
      res = ev.evaluateAll(transformedData)
      _ = metricsPath.foreach(spark.sparkContext.parallelize(Seq(res.toJson()), 1).saveAsTextFile(_))
    } yield res

    // Pick which features to return (always force the key and result features to be included)
    val featuresToKeep: Array[String] = (keepRawFeatures, keepIntermediateFeatures) match {
      case (true, true) => Array.empty
      case (true, false) => (rawFeatures ++ resultFeatures).map(_.name) :+ KeyFieldName
      case (false, true) => stages.map(_.getOutputFeatureName) :+ KeyFieldName
      case (false, false) => resultFeatures.map(_.name) :+ KeyFieldName
    }
    val scores = featuresToKeep.distinct.toList.sorted match {
      case head :: tail => transformedData.select(head, tail: _*)
      case _ => transformedData
    }
    if (log.isTraceEnabled) {
      log.trace("Scores dataframe schema:\n{}", scores.schema.treeString)
      log.trace("Scores dataframe plans:\n")
      scores.explain(extended = true)
    }

    // Persist the scores if needed
    if (persistScores) scores.persist()

    // Save the scores if a path was provided
    path.foreach(scores.saveAvro(_))

    scores -> metrics
  }

}

case object OpWorkflowModel {

  val KeepRawFeatures = false
  val KeepIntermediateFeatures = false
  val PersistEveryKStages = 5
  val PersistScores = true

}
