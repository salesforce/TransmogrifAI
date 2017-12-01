/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.UID
import com.salesforce.op.evaluators.{OpClassificationEvaluatorBase, OpEvaluatorBase, OpMultiClassificationEvaluatorBase}
import com.salesforce.op.features.{FeatureLike, TransientFeature}
import com.salesforce.op.features.types._
import com.salesforce.op.stages._
import com.salesforce.op.stages.impl.classification.ProbabilisticClassifierType.ProbClassifier
import com.salesforce.op.stages.impl.selector._
import com.salesforce.op.stages.impl.tuning.{OpValidator, _}
import com.salesforce.op.stages.sparkwrappers.generic.{SwQuaternaryTransformer, SwTernaryTransformer}
import org.apache.spark.ml.param._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Abstract Model Selector that selects classifier model
 *
 * @param validator Cross Validation or Train Validation Split
 * @param splitter  instance that will balance and split the data
 * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation.
 * @param uid
 */
private[op] abstract class ClassificationModelSelector
(
  val validator: OpValidator[ProbClassifier],
  val splitter: Option[Splitter],
  val trainTestEvaluators: Seq[OpEvaluatorBase[_]],
  val uid: String = UID[ClassificationModelSelector]
) extends Estimator[BestModel]
  with OpPipelineStage2to3[RealNN, OPVector, RealNN, OPVector, OPVector]
  with SelectorClassifiers {

  private[classification] val stage1uid: String = UID[Stage1ClassificationModelSelector]
  private[classification] val stage2uid: String = UID[SwTernaryTransformer[_, _, _, _, _]]
  private[classification] val stage3uid: String = UID[SwQuaternaryTransformer[_, _, _, _, _, _]]

  private[op] val stage1: Stage1ClassificationModelSelector

  private[op] lazy val stage2 = new SwTernaryTransformer[RealNN, OPVector, RealNN, OPVector, SelectedModel](
    inputParam1Name = inputParam1Name,
    inputParam2Name = inputParam2Name,
    inputParam3Name = stage1OperationName,
    outputParamName = outputParam2Name,
    operationName = stage2OperationName,
    sparkMlStageIn = None,
    uid = stage2uid
  )

  private[op] lazy val stage3 = new SwQuaternaryTransformer[RealNN, OPVector, RealNN, OPVector, OPVector,
    SelectedModel](
    inputParam1Name = inputParam1Name,
    inputParam2Name = inputParam2Name,
    inputParam3Name = stage1OperationName,
    inputParam4Name = stage2OperationName,
    outputParamName = outputParam3Name,
    operationName = stage3OperationName,
    sparkMlStageIn = None,
    uid = stage3uid
  )

  final override protected def subStage: Option[Stage1ClassificationModelSelector] = Option(stage1)

  private lazy val stg1out = stage1.getOutput()
  private lazy val stg2out = stage2.getOutput()
  private lazy val stg3out = stage3.getOutput()


  // set substage inputs
  override def onSetInput(): Unit = {
    stage1.setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[OPVector])
    stage2.setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[OPVector], stg1out)
    stage3.setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[OPVector], stg1out, stg2out)
  }

  final override def fit(dataset: Dataset[_]): BestModel = {
    val model = stage1.fit(dataset)
    setMetadata(model.getMetadata())
    new BestModel(model, stage2, stage3, uid)
      .setParent(this)
      .setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[OPVector])
      .setMetadata(model.getMetadata())
  }

  final override def getOutput(): (FeatureLike[RealNN], FeatureLike[OPVector], FeatureLike[OPVector]) = {
    (stg1out, stg2out, stg3out)
  }
}

/**
 * Wrapper for the best model returned by ClassificationModelSelector
 *
 * @param stage1 first wrapping stage for output one (this is the only stage that actually does anything)
 * @param stage2 second stage - dummy for generating second output
 * @param stage3 third stage - dummy for generating third output
 */
private[op] class BestModel
(
  val stage1: SelectedModel,
  val stage2: SwTernaryTransformer[RealNN, OPVector, RealNN, OPVector, SelectedModel],
  val stage3: SwQuaternaryTransformer[RealNN, OPVector, RealNN, OPVector, OPVector, SelectedModel],
  val uid: String = UID[BestModel]
) extends Model[BestModel]
  with OpPipelineStage2to3[RealNN, OPVector, RealNN, OPVector, OPVector]
  with Stage3ParamNamesBase {

  override def transform(dataset: Dataset[_]): DataFrame = {
    stage1.transform(dataset)
  }

  override def getOutput(): (FeatureLike[RealNN], FeatureLike[OPVector], FeatureLike[OPVector]) = {
    (stage1.getOutput(), stage2.getOutput(), stage3.getOutput())
  }

  /**
   * Get parameter of best model.
   *
   * @group getParam
   */
  def getParams: ParamMap = stage1.getSparkMlStage().get.extractParamMap()
}

/**
 * Abstract Stage 1 for Classification Model Selectors
 *
 * @param validator validator used for the selection. It can be either CrossValidation or TrainValidationSplit
 * @param splitter  to split and/or balance the dataset
 * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation.
 * @param uid
 */
private[classification] abstract class Stage1ClassificationModelSelector
(
  validator: OpValidator[ProbClassifier],
  splitter: Option[Splitter],
  trainTestEvaluators: Seq[OpClassificationEvaluatorBase[_]],
  uid: String = UID[Stage1ClassificationModelSelector],
  val stage2uid: String = UID[SwTernaryTransformer[_, _, _, _, _]],
  val stage3uid: String = UID[SwQuaternaryTransformer[_, _, _, _, _, _]]
) extends ModelSelectorBase[ProbClassifier](validator = validator, splitter = splitter,
  trainTestEvaluators = trainTestEvaluators, uid = uid) with SelectorClassifiers {

  lazy val rawPredictionColName: String = outputsColNamesMap(outputParam2Name)
  lazy val probabilityColName: String = outputsColNamesMap(outputParam3Name)

  final override protected def getOutputsColNamesMap(
    f1: TransientFeature, f2: TransientFeature
  ): Map[String, String] = {
    Map(outputParam1Name -> makeOutputNameFromStageId[RealNN](uid, Seq(f1, f2)),
      outputParam2Name -> makeOutputNameFromStageId[OPVector](stage2uid, Seq(f1, f2), 2),
      outputParam3Name -> makeOutputNameFromStageId[OPVector](stage3uid, Seq(f1, f2), 3)
    )
  }

  final override protected def getModelInfo: Seq[ModelInfo[ProbClassifier]] = modelInfo
}
