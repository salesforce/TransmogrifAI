/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import com.salesforce.op.evaluators.{EvaluationMetrics, OpEvaluatorBase}
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.features.{FeatureLike, OPFeature}
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.stages.{OPStage, OpPipelineStage, OpTransformer}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.json4s.JValue
import org.json4s.JsonAST.{JField, JObject}
import org.json4s.jackson.JsonMethods.{pretty, render}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


/**
 * Workflow model is a container and executor for the sequence of transformations that have been fit to the data
 * to produce the desired output features
 *
 * @param uid unique identifier for this workflow model
 * @param trainingParams params that were used during model training
 */
class OpWorkflowModel(val uid: String = UID[OpWorkflowModel], val trainingParams: OpParams) extends OpWorkflowCore {

  /**
   * The parent workflow that produced this model.
   */
  @transient var parent: OpWorkflow = _

  /**
   * Sets the parent of this model.
   */
  def setParent(parent: OpWorkflow): OpWorkflowModel = {
    this.parent = parent
    this.asInstanceOf[OpWorkflowModel]
  }

  /**
   * Get the un-fit workflow that created this workflow model
   *
   * @return OpWorkflow
   */
  def getParent(): OpWorkflow = {
    this.parent
  }

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


  /**
   * Returns a dataframe containing all the columns generated up to the feature input
   *
   * @param feature input feature to compute up to
   * @throws IllegalArgumentException if a feature is not part of this workflow
   * @return Dataframe containing columns corresponding to all of the features generated before the feature given
   */
  def computeDataUpTo(feature: OPFeature)(implicit spark: SparkSession): DataFrame = {
    computeDataUpTo(stopStage = findOriginStageId(feature), fitted = true)
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
        ModelInsights.extractFromStages(modelStages, rawFeatures, trainingParams)
    }
  }

  /**
   * Pulls all summary metadata off of transformers
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
   * Pulls all summary metadata off of transformers and puts them in a pretty json string
   *
   * @return string summary
   */
  def summary(): String = pretty(render(summaryJson()))

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
    keepRawFeatures: Boolean = OpWorkflowModel.keepRawFeatures,
    keepIntermediateFeatures: Boolean = OpWorkflowModel.keepIntermediateFeatures,
    persistEveryKStages: Int = OpWorkflowModel.persistEveryKStages,
    persistScores: Boolean = OpWorkflowModel.persistScores
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
    keepRawFeatures: Boolean = OpWorkflowModel.keepRawFeatures,
    keepIntermediateFeatures: Boolean = OpWorkflowModel.keepIntermediateFeatures,
    persistEveryKStages: Int = OpWorkflowModel.persistEveryKStages,
    persistScores: Boolean = OpWorkflowModel.persistScores,
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
    keepRawFeatures: Boolean = OpWorkflowModel.keepRawFeatures,
    keepIntermediateFeatures: Boolean = OpWorkflowModel.keepIntermediateFeatures,
    persistEveryKStages: Int = OpWorkflowModel.persistEveryKStages,
    persistScores: Boolean = OpWorkflowModel.persistScores,
    evaluator: Option[OpEvaluatorBase[_ <: EvaluationMetrics]] = None,
    metricsPath: Option[String] = None
  )(implicit spark: SparkSession): Option[String] => (DataFrame, Option[EvaluationMetrics]) = {
    require(persistEveryKStages >= 1, s"persistEveryKStages value of $persistEveryKStages is invalid must be >= 1")

    // TODO: replace 'stages' with 'stagesDag'. (is a breaking change for serialization, but would simplify scoreFn)

    // Pre-compute transformations dag
    val dag = computeStagesDAG(resultFeatures)

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
      case (false, true) => stages.map(_.outputName) :+ KeyFieldName
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

  val keepRawFeatures = false
  val keepIntermediateFeatures = false
  val persistEveryKStages = 5
  val persistScores = true

}
