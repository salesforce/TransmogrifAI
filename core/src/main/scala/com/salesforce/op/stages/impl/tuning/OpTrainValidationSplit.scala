/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.evaluators.OpEvaluatorBase
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{TrainValidationSplitModel, OpTrainValidationSplit => LeakageFreeTrainValidationSplit}
import org.apache.spark.sql.types.MetadataBuilder


private[impl] class OpTrainValidationSplit[E <: Estimator[_]]
(
  val trainRatio: Double = ValidatorParamDefaults.TrainRatio,
  val seed: Long = ValidatorParamDefaults.Seed,
  val evaluator: OpEvaluatorBase[_]
) extends OpValidator[E] {

  override def validationParams: Map[String, Any] = Map(
    "metric" -> evaluator.name,
    "trainRatio" -> trainRatio,
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
    new LeakageFreeTrainValidationSplit()
      .setSeed(seed)
      .setEstimator(estimator)
      .setEstimatorParamMaps(paramGrids)
      .setTrainRatio(trainRatio)
      .setEvaluator(evaluator.set(labelCol, label))
      .setHasLeakage(hasLeakage)
  }

  final override private[op] def bestModel[M <: Model[M]](
    allModelsStr: String,
    fittedModels: Seq[(Any, Array[ParamMap])],
    meta: MetadataBuilder): BestModel[M] = {
    log.info(
      "Model Selection over {} with a Train Validation Split with a train ratio of {} evaluated with the {} metric",
      allModelsStr, trainRatio.toString, evaluator.name
    )
    val tvsFittedModelsWithName = fittedModels.map { case (model, paramGrids) =>
      val tvsMod = model.asInstanceOf[TrainValidationSplitModel]
      val bestModelName = updateBestModelMetadata(meta, paramGrids, tvsMod.bestModel.uid,
        tvsMod.bestModel.extractParamMap(), tvsMod.validationMetrics)
      (tvsMod, bestModelName)
    }
    val newMeta = new MetadataBuilder().putMetadata(ModelSelectorBaseNames.TrainValSplitResults, meta.build())

    val (bestTvsFittedModel, bestModelName) =
      if (evaluator.isLargerBetter) tvsFittedModelsWithName.maxBy(_._1.validationMetrics.max)
      else tvsFittedModelsWithName.minBy(_._1.validationMetrics.min)

    BestModel(
      name = bestModelName,
      model = bestTvsFittedModel.bestModel.asInstanceOf[Model[M]],
      metadata = Option(newMeta)
    )
  }
}
