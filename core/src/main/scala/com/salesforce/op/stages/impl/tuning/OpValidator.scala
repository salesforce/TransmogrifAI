/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.evaluators.OpEvaluatorBase
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.types.MetadataBuilder
import org.slf4j.LoggerFactory

/**
 * Best Model container
 * @param name the name of the best model
 * @param model best trained model
 * @param metadata optional metadata
 * @tparam M model type
 */
case class BestModel[M <: Model[M]](name: String, model: Model[M], metadata: Option[MetadataBuilder] = None)

/**
 * Abstract class for Validator: Cross Validation or Train Validation Split
 */
private[impl] trait OpValidator[E <: Estimator[_]] extends Serializable {

  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  val seed: Long
  val evaluator: OpEvaluatorBase[_]

  /**
   * Validation parameters map
   */
  def validationParams: Map[String, Any]

  /**
   * Function that performs the model selection
   *
   * @param estimator  estimator used
   * @param paramGrids param grids
   * @param label
   * @return estimator
   */
  private[op] def validate(
    estimator: E,
    paramGrids: Array[ParamMap],
    label: String
  ): Estimator[_]

  /**
   * get the best model and the metadata with with the validator params
   * @param allModelsStr name of all the different models
   * @param fittedModels sequence of all models with their params
   * @param meta metadata
   * @return best model
   */
  private[op] def bestModel[M <: Model[M]](
    allModelsStr: String,
    fittedModels: Seq[(Any, Array[ParamMap])],
    meta: MetadataBuilder
  ): BestModel[M]

  /**
   * Update metadata during model selection and return best model name
   * @param metadataBuilder
   * @param paramGrids
   * @param modelUid
   * @param modelParams
   * @param modelMetrics
   * @return best model name
   */
  private[op] def updateBestModelMetadata(
    metadataBuilder: MetadataBuilder,
    paramGrids: Array[ParamMap],
    modelUid: String,
    modelParams: ParamMap,
    modelMetrics: Array[Double]
  ): String = {
    def getModelName(index: Int) = s"${modelUid}_$index"

    for {((paramGrid, met), ind) <- paramGrids.zip(modelMetrics).zipWithIndex} {
      val paramMetBuilder = new MetadataBuilder()
      paramMetBuilder.putString(evaluator.name, met.toString)
      // put in all model params from the winner
      modelParams.toSeq.foreach(p => paramMetBuilder.putString(p.param.name, p.value.toString))
      // override with param map values if they exists
      paramGrid.toSeq.foreach(p => paramMetBuilder.putString(p.param.name, p.value.toString))
      metadataBuilder.putMetadata(getModelName(ind), paramMetBuilder.build())
    }
    val bestModelIndex =
      if (evaluator.isLargerBetter) modelMetrics.zipWithIndex.maxBy(_._1)._2
      else modelMetrics.zipWithIndex.minBy(_._1)._2

    getModelName(bestModelIndex)
  }
}

object ValidatorParamDefaults {
  // scalastyle:off method.name
  def Seed: Long = util.Random.nextLong

  val labelCol = "labelCol"
  val NumFolds = 3
  val TrainRatio = 0.75
}

