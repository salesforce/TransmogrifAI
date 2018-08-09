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

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.evaluators.{EvaluationMetrics, _}
import com.salesforce.op.features.types._
import com.salesforce.op.stages._
import com.salesforce.op.stages.base.binary.OpTransformer2
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, Dataset}


case object ModelSelectorBase {
  val TrainValSplitResults = "trainValidationSplitResults"
  val CrossValResults = "crossValidationResults"
  val TrainingEval = "trainingSetEvaluationResults"
  val HoldOutEval = "testSetEvaluationResults"
  val ResampleValues = "resamplingValues"
  val CuttValues = "cuttValues"
  val BestModelUid = "bestModelUID"
  val BestModelName = "bestModelName"
  val Positive = "positiveLabels"
  val Negative = "negativeLabels"
  val Desired = "desiredFraction"
  val UpSample = "upSamplingFraction"
  val DownSample = "downSamplingFraction"
  val idColName = "rowId"
  val LabelsKept = "labelsKept"
  val LabelsDropped = "labelsDropped"

  type ModelType = Model[_ <: Model[_]] with OpTransformer2[RealNN, OPVector, Prediction]
  type EstimatorType = Estimator[_ <: Model[_]] with OpPipelineStage2[RealNN, OPVector, Prediction]
}

/**
 * Trait to mix into Estimators that you wish to work with cross validation and training data holdout
 */
private[op] trait HasEval {

  def evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]]

  protected[op] def outputsColNamesMap: Map[String, String]
  protected[op] def labelColName: String

  protected[op] def fullPredictionColName: Option[String] = outputsColNamesMap.get(StageParamNames.outputParamName)
  protected[op] def predictionColName: Option[String] = outputsColNamesMap.get(StageParamNames.outputParam1Name)
  protected[op] def rawPredictionColName: Option[String] = outputsColNamesMap.get(StageParamNames.outputParam2Name)
  protected[op] def probabilityColName: Option[String] = outputsColNamesMap.get(StageParamNames.outputParam3Name)

  /**
   * Function that evaluates the selected model on the test set
   *
   * @param data              data transformed by best model
   * @return EvaluationMetrics
   */
  protected def evaluate(
    data: Dataset[_]
  ): EvaluationMetrics = {
    data.persist()
    val metricsMap = evaluators.map {
      case evaluator: OpClassificationEvaluatorBase[_] =>
        evaluator.setLabelCol(labelColName)
        fullPredictionColName.foreach(evaluator.setFullPredictionCol)
        predictionColName.foreach(evaluator.setPredictionCol)
        rawPredictionColName.foreach(evaluator.setRawPredictionCol)
        probabilityColName.foreach(evaluator.setProbabilityCol)
        evaluator.name.humanFriendlyName -> evaluator.evaluateAll(data)
      case evaluator: OpRegressionEvaluatorBase[_] =>
        evaluator.setLabelCol(labelColName)
        fullPredictionColName.foreach(evaluator.setFullPredictionCol)
        predictionColName.foreach(evaluator.setPredictionCol)
        evaluator.name.humanFriendlyName -> evaluator.evaluateAll(data)
      case evaluator => throw new RuntimeException(s"Evaluator $evaluator is not supported")
    }.toMap
    data.unpersist()
    MultiMetrics(metricsMap)
  }

}

/**
 * Trait to mix into Model that you wish to work with cross validation and training data holdout
 */
private[op] trait HasTestEval extends HasEval {

  self: Model[_] with OpPipelineStageBase =>

  // TODO may want to allow others to use this eventually but then need to get evaluators on deser
  /**
   * Evaluation function for workflow to call on test dataset
   * @param data transformed data
   * @return transforms data and sets evaluation metadata
   */
  private[op] def evaluateModel(data: Dataset[_]): DataFrame = {
    val scored = transform(data)
    val metrics = evaluate(scored)
    val metadata = ModelSelectorSummary.fromMetadata(getMetadata().getSummaryMetadata())
      .copy(holdoutEvaluation = Option(metrics))
    setMetadata(metadata.toMetadata().toSummaryMetadata())
    scored
  }
}
