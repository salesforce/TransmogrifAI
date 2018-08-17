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

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.UID
import com.salesforce.op.evaluators.{EvaluationMetrics, OpClassificationEvaluatorBase, OpEvaluatorBase}
import com.salesforce.op.features.{FeatureLike, TransientFeature}
import com.salesforce.op.features.types._
import com.salesforce.op.stages._
import com.salesforce.op.stages.impl.classification.ProbabilisticClassifierType._
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
 * @param evaluators List of evaluators applied on training + holdout data for evaluation.
 * @param uid
 */
private[op] abstract class ClassificationModelSelector
(
  val validator: OpValidator[ProbClassifierModel, ProbClassifier],
  val splitter: Option[Splitter],
  val evaluators: Seq[OpEvaluatorBase[_]],
  val uid: String = UID[ClassificationModelSelector]
) extends Estimator[BestModel]
  with OpPipelineStage2to3[RealNN, OPVector, RealNN, OPVector, OPVector]
  with SelectorClassifiers {

  private[classification] val stage1uid: String = UID[Stage1ClassificationModelSelector]
  private[classification] val stage2uid: String = UID[SwTernaryTransformer[_, _, _, _, _]]
  private[classification] val stage3uid: String = UID[SwQuaternaryTransformer[_, _, _, _, _, _]]

  private[op] val stage1: Stage1ClassificationModelSelector

  private[op] lazy val stage2 = new SwTernaryTransformer[RealNN, OPVector, RealNN, OPVector, SelectedModel](
    inputParam1Name = StageParamNames.inputParam1Name,
    inputParam2Name = StageParamNames.inputParam2Name,
    inputParam3Name = stage1OperationName,
    outputParamName = StageParamNames.outputParam2Name,
    operationName = stage2OperationName,
    sparkMlStageIn = None,
    uid = stage2uid
  )

  private[op] lazy val stage3 = new SwQuaternaryTransformer[RealNN, OPVector, RealNN, OPVector, OPVector,
    SelectedModel](
    inputParam1Name = StageParamNames.inputParam1Name,
    inputParam2Name = StageParamNames.inputParam2Name,
    inputParam3Name = stage1OperationName,
    inputParam4Name = stage2OperationName,
    outputParamName = StageParamNames.outputParam3Name,
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
final class BestModel private[op]
(
  val stage1: SelectedModel,
  val stage2: SwTernaryTransformer[RealNN, OPVector, RealNN, OPVector, SelectedModel],
  val stage3: SwQuaternaryTransformer[RealNN, OPVector, RealNN, OPVector, OPVector, SelectedModel],
  val uid: String = UID[BestModel]
) extends Model[BestModel] with OpPipelineStage2to3[RealNN, OPVector, RealNN, OPVector, OPVector]
  with StageOperationName with HasTestEval {

  override def evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = stage1.evaluators
  override protected[op] def outputsColNamesMap: Map[String, String] = stage1.outputsColNamesMap
  override protected[op] def labelColName: String = stage1.labelColName

  override private[op] def evaluateModel(data: Dataset[_]) = {
    val transformed = stage1.evaluateModel(data)
    setMetadata(stage1.getMetadata())
    transformed
  }

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
 * @param evaluators List of evaluators applied on training + holdout data for evaluation.
 * @param uid
 */
private[classification] abstract class Stage1ClassificationModelSelector
(
  validator: OpValidator[ProbClassifierModel, ProbClassifier],
  splitter: Option[Splitter],
  evaluators: Seq[OpClassificationEvaluatorBase[_ <: com.salesforce.op.evaluators.EvaluationMetrics]],
  uid: String = UID[Stage1ClassificationModelSelector],
  val stage2uid: String = UID[SwTernaryTransformer[_, _, _, _, _]],
  val stage3uid: String = UID[SwQuaternaryTransformer[_, _, _, _, _, _]]
) extends ModelSelectorBase[ProbClassifierModel, ProbClassifier](validator = validator, splitter = splitter,
  evaluators = evaluators, uid = uid) with SelectorClassifiers {

  final override protected def getOutputsColNamesMap(
    f1: TransientFeature, f2: TransientFeature
  ): Map[String, String] = {
    Map(StageParamNames.outputParam1Name -> makeOutputNameFromStageId[RealNN](uid, Seq(f1, f2)),
      StageParamNames.outputParam2Name -> makeOutputNameFromStageId[OPVector](stage2uid, Seq(f1, f2), 2),
      StageParamNames.outputParam3Name -> makeOutputNameFromStageId[OPVector](stage3uid, Seq(f1, f2), 3)
    )
  }

}
