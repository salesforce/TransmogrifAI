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

import com.salesforce.op.UID
import com.salesforce.op.evaluators.{EvaluationMetrics, _}
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataFrameFieldNames
import com.salesforce.op.stages._
import com.salesforce.op.stages.base.binary.OpTransformer2
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.RichParamMap._
import com.salesforce.op.utils.stages.FitStagesUtil._
import org.apache.spark.ml.param._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.reflect.runtime.universe._


/**
 * Model selector class which will take in models with label (RealNN) and feature (OPVector) inputs and a Prediction
 * output. It will prepare data as specified in the splitter and perform validation as specified in the validator to
 * select the best model and hyperparameters based in the evaluation metric specified.
 * @param validator Performs training split of cross validation to select the best model
 * @param splitter prepares and splits the data for training
 * @param models models to try and the hyperparameters to try with each model type
 * @param evaluators final evaluators to use after model selection for holdout and training data
 * @param uid uid of stage
 * @param operationName name of stage
 * @param tti1 type tag for RealNN
 * @param tti2 type tag for OPVector
 * @param tto type tag for Prediction
 * @param ttov type tag for Prediction internal map
 * @tparam M type of model returned from estimator
 * @tparam E type of estimators comparing for model selection
 */
private[op] class ModelSelector[M <: Model[_] with OpTransformer2[RealNN, OPVector, Prediction],
E <: Estimator[_] with OpPipelineStage2[RealNN, OPVector, Prediction]]
(
  val validator: OpValidator[M, E],
  val splitter: Option[Splitter],
  val models: Seq[(E, Array[ParamMap])],
  val evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]],
  val uid: String = UID[ModelSelector[M, E]],
  val operationName: String = "modelSelection"
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends Estimator[SelectedModel]
  with OpPipelineStage2[RealNN, OPVector, Prediction] with HasEval {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  @transient private[op] var bestEstimator: Option[BestEstimator[E]] = None
  @transient private val modelsUse = models.map{case (e, p) =>
    val est = e.setOutputFeatureName(getOutputFeatureName)
    val par = if (p.isEmpty) Array(new ParamMap) else p
    est -> par
  }

  /**
   * Find best estimator with validation on a workflow level. Executed when workflow level Cross Validation is on
   * (see [[com.salesforce.op.OpWorkflow.withWorkflowCV]])
   *
   * @param data                data to validate
   * @param dag                 dag done inside the Cross-validation/Train-validation split
   * @param persistEveryKStages frequency of persisting the DAG's stages
   * @param spark               Spark Session
   * @return Updated Model Selector with best model along with best paramMap
   */
  protected[op] def findBestEstimator(data: Dataset[_], dag: StagesDAG, persistEveryKStages: Int = 0)
    (implicit spark: SparkSession): Unit = {

    val theBestEstimator = validator.validate(modelInfo = modelsUse, dataset = data,
      label = in1.name, features = in2.name, dag = Option(dag), splitter = splitter,
      stratifyCondition = validator.isClassification
    )

    bestEstimator = Option(theBestEstimator)
  }

  lazy val labelColName: String = in1.name

  override protected[op] def outputsColNamesMap: Map[String, String] =
    Map(StageParamNames.outputParamName -> getOutputFeatureName)

  /**
   * Splits the data into training test and test set, balances the training set and selects the best model
   * Tests the model on the test set and prints results
   *
   * @param dataset
   * @return best model
   */
  final override def fit(dataset: Dataset[_]): SelectedModel = {

    implicit val spark = dataset.sparkSession

    val datasetWithID =
      if (dataset.columns.contains(DataFrameFieldNames.KeyFieldName)) {
        dataset.select(in1.name, in2.name, DataFrameFieldNames.KeyFieldName)
      } else {
        dataset.select(in1.name, in2.name)
          .withColumn(ModelSelectorBase.idColName, monotonically_increasing_id())
      }
    require(!datasetWithID.isEmpty, "Dataset cannot be empty")

    val ModelData(trainData, met) = splitter match {
      case Some(spltr) => spltr.prepare(datasetWithID)
      case None => ModelData(datasetWithID, None)
    }

    val BestEstimator(name, estimator, summary) = bestEstimator.getOrElse{
      setInputSchema(dataset.schema).transformSchema(dataset.schema)
      val best = validator
        .validate(modelInfo = modelsUse, dataset = trainData, label = in1.name, features = in2.name)
      bestEstimator = Some(best)
      best
    }

    val bestModel = estimator.fit(trainData).asInstanceOf[M]
    val bestEst = bestModel.parent
    log.info(s"Selected model : ${bestEst.getClass.getSimpleName}")
    log.info(s"With parameters : ${bestEst.extractParamMap()}")

    // set input and output params
    outputsColNamesMap.foreach { case (pname, pvalue) => bestModel.set(bestModel.getParam(pname), pvalue) }

    // get eval results for metadata
    val trainingEval = evaluate(bestModel.transform(trainData))

    val metadataSummary = ModelSelectorSummary(
      validationType = ValidationType.fromValidator(validator),
      validationParameters = validator.getParams(),
      dataPrepParameters = splitter.map(_.extractParamMap().getAsMap()).getOrElse(Map()),
      dataPrepResults = met,
      evaluationMetric = validator.evaluator.name,
      problemType = ProblemType.fromEvalMetrics(trainingEval),
      bestModelUID = estimator.uid,
      bestModelName = name,
      bestModelType = estimator.getClass.getSimpleName,
      validationResults = summary,
      trainEvaluation = trainingEval,
      holdoutEvaluation = None
    )

    setMetadata(metadataSummary.toMetadata().toSummaryMetadata())

    new SelectedModel(
      bestModel.asInstanceOf[ModelSelectorBase.ModelType],
      outputsColNamesMap,
      uid,
      operationName
    )
      .setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[OPVector])
      .setParent(this)
      .setMetadata(getMetadata())
      .setOutputFeatureName(getOutputFeatureName)
      .setEvaluators(evaluators)
  }

}

/**
 * Returned wrapped best model from model selector estimator
 * @param modelStageIn the best model from those tried in the estimator
 * @param outputsColNamesMap a map of the output names for the estimator
 * @param uid uid of the stage
 * @param operationName name of stage
 * @param tti1 type tag for RealNN
 * @param tti2 type tag for OPVector
 * @param tto type tag for Prediction
 * @param ttov type tag for Prediction internal map
 */
final class SelectedModel private[op]
(
  val modelStageIn: ModelSelectorBase.ModelType,
  val outputsColNamesMap: Map[String, String],
  val uid: String,
  val operationName: String
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends Model[SelectedModel] with OpTransformer2[RealNN, OPVector, Prediction] with HasTestEval {

  override def transformFn: (RealNN, OPVector) => Prediction = modelStageIn.transformFn

  lazy val labelColName: String = in1.name

  // TODO this is lost on serialization if we want to use the eval method here in eval runs as well as training
  // need to pass evaluators from origin stage to deserialized in OpPipelineStageReader.loadModel
  @transient private var evaluatorList: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty
  def setEvaluators(ev: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]]): this.type = {
    evaluatorList = ev
    this
  }
  override def evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = evaluatorList

}
