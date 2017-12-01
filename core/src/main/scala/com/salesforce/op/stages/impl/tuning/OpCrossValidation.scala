/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.evaluators.OpEvaluatorBase
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidatorModel, OpCrossValidator}
import org.apache.spark.sql.types.MetadataBuilder
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames



private[impl] class OpCrossValidation[E <: Estimator[_]]
(
  val numFolds: Int = ValidatorParamDefaults.NumFolds,
  val seed: Long = ValidatorParamDefaults.Seed,
  val evaluator: OpEvaluatorBase[_]
) extends OpValidator[E]
  {

  override def validationParams: Map[String, Any] = Map(
    "metric" -> evaluator.name,
    "numFolds" -> numFolds,
    "seed" -> seed
  )

  private[op] def validate(
    estimator: E,
    paramGrids: Array[ParamMap],
    label: String,
    hasLeakage: Boolean
  ): Estimator[_] = {

    // get param that stores the label column
    val labelCol = evaluator.getParam(ValidatorParamDefaults.labelCol)
    new OpCrossValidator()
      .setSeed(seed)
      .setEstimator(estimator)
      .setEstimatorParamMaps(paramGrids)
      .setNumFolds(numFolds)
      .setEvaluator(evaluator.set(labelCol, label))
      .setHasLeakage(hasLeakage)
  }

  final override private[op] def bestModel[M <: Model[M]](
    allModelsStr: String,
    fittedModels: Seq[(Any, Array[ParamMap])],
    meta: MetadataBuilder): BestModel[M] = {
    log.info(
      "Model Selection over {} with a Cross Validation of {} folds evaluated with the {} metric",
      allModelsStr, numFolds.toString, evaluator.name
    )
    val cvFittedModels = fittedModels.map { case (model, paramGrids) =>
      val cvMod = model.asInstanceOf[CrossValidatorModel]
      val bestModelName = updateBestModelMetadata(meta, paramGrids, cvMod.bestModel.uid,
        cvMod.bestModel.extractParamMap(), cvMod.avgMetrics)
      (cvMod, bestModelName)
    }
    val newMeta = new MetadataBuilder().putMetadata(ModelSelectorBaseNames.CrossValResults, meta.build())

    val (bestCvFittedModel, bestModelName) =
      if (evaluator.isLargerBetter) cvFittedModels.maxBy(_._1.avgMetrics.max)
      else cvFittedModels.minBy(_._1.avgMetrics.min)

    BestModel(
      name = bestModelName,
      model = bestCvFittedModel.bestModel.asInstanceOf[Model[M]],
      metadata = Option(newMeta)
    )
  }
}
