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

import com.salesforce.op.evaluators.{EvaluationMetrics, OpEvaluatorBase}
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.features.{Feature, FeatureLike, OPFeature}
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.stages.{OPStage, OpPipelineStage, OpTransformer}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.{JobGroupUtil, OpStep}
import com.salesforce.op.utils.stages.FitStagesUtil
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.json4s.JValue
import org.json4s.JsonAST.{JField, JObject}
import org.json4s.jackson.JsonMethods.{pretty, render}

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

  protected[op] def setBlocklist(features: Array[OPFeature]): this.type = {
    blocklistedFeatures = features
    this
  }

  protected[op] def setBlocklistMapKeys(mapKeys: Map[String, Set[String]]): this.type = {
    blocklistedMapKeys = mapKeys
    this
  }

  /**
   * Used to generate dataframe from reader and raw features list
   *
   * @return Dataframe with all the features generated + persisted
   */
  protected def generateRawData()(implicit spark: SparkSession): DataFrame = {
    JobGroupUtil.withJobGroup(OpStep.DataReadingAndFiltering) {
      require(reader.nonEmpty, "Data reader must be set")
      checkFeatures()
      reader.get.generateDataFrame(rawFeatures, parameters).persist() // don't want to redo this
    }
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
    val allFeatures = getRawFeatures() ++ getBlocklist() ++ getStages().map(_.getOutput())
    features.map { f =>
      allFeatures.find(_.sameOrigin(f))
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
        ModelInsights.extractFromStages(modelStages, rawFeatures, trainingParams,
          getBlocklist(), getBlocklistMapKeys(), getRawFeatureFilterResults())
    }
  }

  /**
   * Extracts all summary metadata from transformers in JSON format
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
   * Extracts all summary metadata from transformers in JSON format
   *
   * @return json string summary
   */
  def summary(): String = pretty(render(summaryJson()))

  /**
   * Generated high level model summary in a compact print friendly format containing:
   * selected model info, model evaluation results and feature correlations/contributions/cramersV values.
   *
   * @param insights model insights to compute the summary against
   * @param topK top K of feature correlations/contributions/cramersV values to print
   * @return high level model summary in a compact print friendly format
   */
  def summaryPretty(
    insights: ModelInsights = modelInsights(
      resultFeatures.find(f => f.isResponse && !f.isRaw).getOrElse(
        throw new IllegalArgumentException("No response feature is defined to compute model insights"))
    ),
    topK: Int = 15
  ): String = insights.prettyPrint(topK)

  /**
   * Save this model to a path
   *
   * @param path      path to save the model
   * @param overwrite should overwrite if the path exists
   * @param modelStagingDir local folder to copy and unpack stored model to for loading
   */
  def save(path: String, overwrite: Boolean = true,
    modelStagingDir: String = WorkflowFileReader.modelStagingDir): Unit = {
    OpWorkflowModelWriter.save(this, path = path, overwrite = overwrite, modelStagingDir)
  }

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
      case (true, true) => Array.empty[String] // keep everything (no `data.select` needed)
      case (true, false) => (rawFeatures ++ resultFeatures).map(_.name) :+ KeyFieldName
      case (false, true) => stages.map(_.getOutputFeatureName) :+ KeyFieldName
      case (false, false) => resultFeatures.map(_.name) :+ KeyFieldName
    }
    val scores = featuresToKeep.distinct match {
      case Array() => transformedData // keep everything (no `data.select` needed)
      case keep =>
        // keep the order of the columns the same when selecting, so the data wont be reshuffled
        val columns = transformedData.columns.filter(keep.contains).map(column)
        transformedData.select(columns: _*)
    }
    if (log.isTraceEnabled) {
      log.trace("Scores dataframe schema:\n{}", scores.schema.treeString)
      log.trace("Scores dataframe plans:\n")
      scores.explain(extended = true)
    }

    // Persist the scores if needed
    if (persistScores) scores.persist()

    // Save the scores if a path was provided
    JobGroupUtil.withJobGroup(OpStep.ResultsSaving) {
      path.foreach(scores.saveAvro(_))
    }

    scores -> metrics
  }

  /**
   * Creates a copy of this [[OpWorkflowModel]] instance
   *
   * @return copy of this [[OpWorkflowModel]] instance
   */
  def copy(): OpWorkflowModel = {
    def copyFeatures(features: Array[OPFeature]): Array[OPFeature] = features.collect { case f: Feature[_] =>
      implicit val tt = f.wtt
      f.copy()
    }
    val copy =
      new OpWorkflowModel(uid = uid, trainingParams = trainingParams.copy())
        .setFeatures(copyFeatures(resultFeatures))
        .setBlocklist(copyFeatures(blocklistedFeatures))
        .setBlocklistMapKeys(blocklistedMapKeys)
        .setRawFeatureFilterResults(rawFeatureFilterResults.copy())
        .setStages(stages.map(_.copy(ParamMap.empty)))
        .setParameters(parameters.copy())

    reader.foreach(copy.setReader)

    if (isWorkflowCV) copy.withWorkflowCV else copy
  }

}

case object OpWorkflowModel {

  val KeepRawFeatures = false
  val KeepIntermediateFeatures = false
  val PersistEveryKStages = 5
  val PersistScores = true

  /**
   * Load a previously trained workflow model from path
   *
   * @param path to the trained workflow model
   * @param asSpark if true will load as spark models if false will load as Mleap stages for spark wrapped stages
   * @param modelStagingDir local folder to copy and unpack stored model to for loading
   * @return workflow model
   */
  def load(
    path: String,
    asSpark: Boolean = true,
    modelStagingDir: String = WorkflowFileReader.modelStagingDir
  ): OpWorkflowModel =
    new OpWorkflowModelReader(None, asSpark).load(path, modelStagingDir)
}
