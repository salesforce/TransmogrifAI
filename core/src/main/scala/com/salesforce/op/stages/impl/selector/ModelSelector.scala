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
import com.salesforce.op.features.types.Prediction.Keys._
import com.salesforce.op.features.types._
import com.salesforce.op.stages._
import com.salesforce.op.stages.base.binary.OpTransformer2
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.ModelType
import com.salesforce.op.stages.impl.tuning.{BestEstimator, _}
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapperModel, SparkModelConverter}
import com.salesforce.op.utils.spark.{JobGroupUtil, OpStep}
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.RichParamMap._
import com.salesforce.op.utils.stages.FitStagesUtil._
import org.apache.spark.ml.param._
import org.apache.spark.ml.{Estimator, Model, PredictionModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
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
  // TODO allow smart modification of these values
  @transient private lazy val modelsUse = models.map{ case (e, p) =>
    val est = e.setOutputFeatureName(getOutputFeatureName)
    val par = if (p.isEmpty) Array(new ParamMap) else p
    est -> par
  }

  lazy val labelColName: String = in1.name

  override protected[op] def outputsColNamesMap: Map[String, String] =
    Map(ModelSelectorNames.outputParamName -> getOutputFeatureName)

  /**
   * Find best estimator with validation on a workflow level. Executed when workflow level Cross Validation is on
   * (see [[com.salesforce.op.OpWorkflow.withWorkflowCV]])
   *
   * @param data                data to validate
   * @param dag                 dag done inside the Cross-validation/Train-validation split
   * @param spark               Spark Session
   * @return Updated Model Selector with best model along with best paramMap
   */
  protected[op] def findBestEstimator(data: DataFrame, dag: Option[StagesDAG] = None)
    (implicit spark: SparkSession): (BestEstimator[E], Option[SplitterSummary], DataFrame) = {

    val PrevalidationVal(splitterSummary, dataOpt) = prepareForValidation(data, labelColName)
    val dataUse = dataOpt.getOrElse(data)

    val summary = validator.validate(modelInfo = modelsUse, dataset = dataUse,
      label = labelColName, features = in2.name, dag = dag, splitter = splitter
    )
    dataUse.unpersist()

    val theBestEstimator = validator.getBestFromVal(summary)

    bestEstimator = Option(theBestEstimator)
    (theBestEstimator, splitterSummary, dataUse)
  }

  protected[op] def findSummary(data: DataFrame, dag: Option[StagesDAG] = None)
    (implicit spark: SparkSession): Array[ValidatedModel[E]] = {

    val PrevalidationVal(_, dataOpt) = prepareForValidation(data, labelColName)
    val dataUse = dataOpt.getOrElse(data)

    val summary = validator.validate(modelInfo = modelsUse, dataset = dataUse,
      label = labelColName, features = in2.name, dag = dag, splitter = splitter
    )
    dataUse.unpersist()

    summary
  }

  private def prepareForValidation(data: DataFrame, labelColName: String): PrevalidationVal = {
    splitter
      .map(_.withLabelColumnName(labelColName).preValidationPrepare(data))
      .getOrElse(PrevalidationVal(None, Option(data)))
  }


  /**
   * Splits the data into training test and test set, balances the training set and selects the best model
   * Tests the model on the test set and prints results
   *
   * @param dataset
   * @return best model
   */
  final override def fit(dataset: Dataset[_]): SelectedModel = {

    implicit val spark = dataset.sparkSession
    JobGroupUtil.setJobGroup(OpStep.CrossValidation)
    setInputSchema(dataset.schema).transformSchema(dataset.schema)
    require(!dataset.isEmpty, "Dataset cannot be empty")
    val data = dataset.select(labelColName, in2.name)
    val (BestEstimator(name, estimator, summary), splitterSummary, datasetWithID) = bestEstimator.map { e =>
      val PrevalidationVal(summary, dataOpt) = prepareForValidation(data, labelColName)
      (e, summary, dataOpt.getOrElse(data))
    }.getOrElse { findBestEstimator(data.toDF()) }

    val preparedData = splitter.map(_.validationPrepare(datasetWithID)).getOrElse(datasetWithID)
    val bestModel = estimator.fit(preparedData).asInstanceOf[M]
//    val bestEst = bestModel.parent
//    log.info(s"Selected model : ${bestEst.getClass.getSimpleName}")
//    log.info(s"With parameters : ${bestEst.extractParamMap()}")

    val modelSummary = findSummary(data.toDF())
    val metricsSum = modelSummary.flatMap{ _.metrics}.sum

    val all = modelSummary.flatMap { case ValidatedModel(model, _, metrics, grids) =>
      grids.zip(metrics).map { case (grid, metric) => {
        val e = model.copy(grid)
        e.fit(preparedData).asInstanceOf[M].asInstanceOf[ModelType] -> metric / metricsSum
      }
      }
    }.sortBy { case (_, m) => if (validator.evaluator.isLargerBetter) -m else m }
    val (allModels, allWeights) = all.unzip[ModelType, Double]

    // set input and output params
    outputsColNamesMap.foreach { case (pname, pvalue) => estimator.set(bestModel.getParam(pname), pvalue) }

    // get eval results for metadata
    val trainingEval = evaluate(bestModel.transform(preparedData))

    val metadataSummary = ModelSelectorSummary(
      validationType = ValidationType.fromValidator(validator),
      validationParameters = validator.getParams(),
      dataPrepParameters = splitter.map(_.extractParamMap().getAsMap()).getOrElse(Map.empty),
      dataPrepResults = splitterSummary,
      evaluationMetric = validator.evaluator.name,
      problemType = ProblemType.fromEvalMetrics(trainingEval),
      bestModelUID = estimator.uid,
      bestModelName = name,
      bestModelType = estimator.getClass.getSimpleName,
      validationResults = summary,
      trainEvaluation = trainingEval,
      holdoutEvaluation = None
    )

    // We skip unsupported metadata values here so the model selector won't break
    // when non standard model parameters are present in param maps
    val meta = metadataSummary.toMetadata(skipUnsupported = true)
    setMetadata(meta.toSummaryMetadata())

    val selectedModel = new SelectedModel(
      allModels.indices.zip(allModels).toMap,
      allWeights,
      outputsColNamesMap,
      uid = uid,
      operationName = operationName
    )
      .setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[OPVector])
      .setParent(this)
      .setMetadata(getMetadata())
      .setOutputFeatureName(getOutputFeatureName)
      .setEvaluators(evaluators)
    // Reset the job group to feature engineering.
    JobGroupUtil.setJobGroup(OpStep.FeatureEngineering)
    selectedModel
  }

}

/**
 * Returned wrapped best model from model selector estimator
 * @param modelStageIn the best model from those tried in the estimator
 * @param weights Weights
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
  val modelStageIn: Map[Int, ModelType],
  val weights: Seq[Double] = Seq(1.0),
  val outputsColNamesMap: Map[String, String],
  val uid: String,
  val operationName: String
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends Model[SelectedModel] with SparkWrapperParams[Model[_]]
  with OpTransformer2[RealNN, OPVector, Prediction] with HasTestEval {

  modelStageIn.map { case (_, model) => model match {
    case m: OpPredictorWrapperModel[_] => setDefault(sparkMlStage, m.getSparkMlStage())
    case m => setDefault(sparkMlStage, Option(m))
  }
  }

//  @transient private lazy val recoveredStages: Array[ModelType] = getSparkMlStage() match {
//    case Some(m: PredictionModel[_, _]) => SparkModelConverter.toOPUnchecked(m).asInstanceOf[ModelType]
//    case Some(m: ModelType@unchecked) => m
//    case m => throw new IllegalArgumentException(s"SparkMlStage in SelectedModel ($m) is of unsupported" +
//      s" type ${m.getClass.getName}")
//  }
  def combine(x: Map[String, Double], y: Map[String, Double]): Map[String, Double] = {
    val keys = x.keys.toSet.union(y.keys.toSet)
  keys.map { case k =>
    val v = x.getOrElse(k, 0.0) + y.getOrElse(k, 0.0)
    k -> v
  }.toMap
  }
  override def transformFn: (RealNN, OPVector) => Prediction = (r, v) => {
    val values = modelStageIn.map { case (i, m) => i -> m.transformFn(r, v) }.zip(weights).map { case ((i, pred), w) =>
      println(s"$v $i $w $pred")
      println(s"Value for $v $i $w  ${pred.value.mapValues(_ * w)}")
      pred.value.mapValues(_ * w)
    }.reduce[Map[String, Double]] {
      case (m1, m2) =>
        println(s"${m1.get("prediction")} ${m2.get("prediction")} ${(combine(m1, m2)).get("prediction")}")
        combine(m1, m2)
    }
    println(s"Values $values")
    val firstPrediction = new Prediction(values)
    val rawPrediction = firstPrediction.rawPrediction
    val probability = firstPrediction.probability

    val newPrediction = if (rawPrediction.length != 0) {
      rawPrediction.zipWithIndex.maxBy(_._1)._2.toDouble
    } else {
      firstPrediction.prediction
    }
    val p = if (rawPrediction.length != 0) {
      Prediction(prediction = newPrediction, rawPrediction = rawPrediction, probability = probability)
    } else {
      Prediction(newPrediction)
    }
    println(s"Final pred for $v ${p}")
    p
  }
    // recoveredStage.transformFn

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
