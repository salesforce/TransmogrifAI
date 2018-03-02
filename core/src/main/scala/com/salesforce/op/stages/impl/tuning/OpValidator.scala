/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.evaluators.OpEvaluatorBase
import com.salesforce.op.stages.impl.selector.ModelInfo
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.MetadataBuilder
import org.slf4j.{Logger, LoggerFactory}

/**
 * Best Model container
 *
 * @param name     the name of the best model
 * @param model    best trained model
 * @param metadata optional metadata
 * @tparam M model type
 */
case class BestModel[M <: Model[_]](name: String, model: M, metadata: Option[MetadataBuilder] = None)


/**
 * Validated Model container
 *
 * @param model     model instance
 * @param bestIndex best metric / grid index
 * @param metrics   all computed metrics
 * @param grids     all param grids
 */
private[tuning] case class ValidatedModel[E <: Estimator[_]]
(
  model: E,
  bestIndex: Int,
  metrics: Array[Double],
  grids: Array[ParamMap]
) {
  /**
   * Best metric (metric at bestIndex)
   */
  def bestMetric: Double = metrics(bestIndex)
  /**
   * Best grid (param grid at bestIndex)
   */
  def bestGrid: ParamMap = grids(bestIndex)
}

/**
 * Abstract class for Validator: Cross Validation or Train Validation Split
 * The model type should be the output of the estimator type but specifying that kills the scala compiler
 */
private[impl] trait OpValidator[M <: Model[_], E <: Estimator[_]] extends Serializable {

  @transient protected lazy val log: Logger = LoggerFactory.getLogger(this.getClass)

  def seed: Long
  def evaluator: OpEvaluatorBase[_]
  def validationName: String

  /**
   * Function that performs the model selection
   *
   * @param modelInfo estimators and grids to validate
   * @param label
   * @param features
   * @param dataset
   * @return estimator
   */
  private[op] def validate(
    modelInfo: Seq[ModelInfo[E]],
    dataset: Dataset[_],
    label: String,
    features: String
  ): BestModel[M]

  /**
   * Get the best model and the metadata with with the validator params
   *
   * @param modelsFit info from validation
   * @param bestModel best fit model
   * @param splitInfo split info for logging
   * @return best model
   */
  private[op] def wrapBestModel(
    modelsFit: Array[ValidatedModel[E]],
    bestModel: M,
    splitInfo: String
  ): BestModel[M] = {
    log.info(
      "Model Selection over {} with {} with {} and the {} metric",
      modelsFit.map(_.model.getClass.getSimpleName).mkString(","), validationName, splitInfo, evaluator.name
    )
    val meta = new MetadataBuilder()
    val cvFittedModels = modelsFit.map(v => updateBestModelMetadata(meta, v) -> v.bestMetric)
    val newMeta = new MetadataBuilder().putMetadata(validationName, meta.build())
    val (bestModelName, _) = if (evaluator.isLargerBetter) cvFittedModels.maxBy(_._2) else cvFittedModels.minBy(_._2)

    BestModel(name = bestModelName, model = bestModel, metadata = Option(newMeta))
  }

  /**
   * Update metadata during model selection and return best model name
   * @return best model name
   */
  private[op] def updateBestModelMetadata(metadataBuilder: MetadataBuilder, v: ValidatedModel[E]): String = {
    val ValidatedModel(model, bestIndex, metrics, grids) = v
    val modelParams = model.extractParamMap()
    def makeModelName(index: Int) = s"${model.uid}_$index"

    for {((paramGrid, met), ind) <- grids.zip(metrics).zipWithIndex} {
      val paramMetBuilder = new MetadataBuilder()
      paramMetBuilder.putString(evaluator.name, met.toString)
      // put in all model params from the winner
      modelParams.toSeq.foreach(p => paramMetBuilder.putString(p.param.name, p.value.toString))
      // override with param map values if they exists
      paramGrid.toSeq.foreach(p => paramMetBuilder.putString(p.param.name, p.value.toString))
      metadataBuilder.putMetadata(makeModelName(ind), paramMetBuilder.build())
    }
    makeModelName(bestIndex)
  }
}

object ValidatorParamDefaults {
  def Seed: Long = util.Random.nextLong // scalastyle:off method.name
  val labelCol = "labelCol"
  val NumFolds = 3
  val TrainRatio = 0.75
}

