/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.UID
import com.salesforce.op.evaluators.{EvaluationMetrics, _}
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataFrameFieldNames
import com.salesforce.op.stages._
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.impl.tuning.SelectorData.LabelFeaturesKey
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.param._
import org.apache.spark.ml.{Estimator, Model, PipelineStage, Transformer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.reflect.runtime.universe._


case object ModelSelectorBaseNames {
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
}

/**
 * Trait to mix into Estimators that you wish to work with cross validation and training data holdout
 */
private[op] trait HasEval {

  def evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]]

  protected[op] def outputsColNamesMap: Map[String, String]
  protected[op] def labelColName: String

  protected[op] def predictionColName: String = outputsColNamesMap(StageParamNames.outputParam1Name)
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
        evaluator.setLabelCol(labelColName).setPredictionCol(predictionColName)
        rawPredictionColName.map(evaluator.setRawPredictionCol)
        probabilityColName.map(evaluator.setProbabilityCol)
        evaluator.name -> evaluator.evaluateAll(data)
      case evaluator: OpRegressionEvaluatorBase[_] =>
        evaluator.setLabelCol(labelColName).setPredictionCol(predictionColName)
        evaluator.name -> evaluator.evaluateAll(data)
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
    val builder = new MetadataBuilder().withMetadata(getMetadata().getSummaryMetadata())
    builder.putMetadata(ModelSelectorBaseNames.HoldOutEval, metrics.toMetadata)
    setMetadata(builder.build().toSummaryMetadata())
    scored
  }
}

/**
 * Factory to implement a Model Selector. Model Selector has only one output, it can be used for the stage 1 of a
 * model selector that outputs prediction, raw prediction and probability
 *
 * @param validator         validator used for the selection. It can be either CrossValidation or TrainValidationSplit
 * @param splitter          to split and/or balance the dataset
 * @param evaluators List of evaluators applied on training + holdout data for evaluation.
 * @param uid
 * @param tti1
 * @param tto
 * @tparam E Type parameter of the estimators used in the model selector
 */
private[op] abstract class ModelSelectorBase[M <: Model[_], E <: Estimator[_]]
(
  val validator: OpValidator[M, E],
  val splitter: Option[Splitter],
  override val evaluators: Seq[OpEvaluatorBase[_ <: com.salesforce.op.evaluators.EvaluationMetrics]],
  val uid: String = UID[ModelSelectorBase[_, _]]
)(
  implicit val tti1: TypeTag[OPVector],
  val tto: TypeTag[RealNN],
  val ttov: TypeTag[RealNN#Value]
) extends Estimator[SelectedModel]
  with OpPipelineStage2[RealNN, OPVector, RealNN]
  with HasEval {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }


  final val operationName: String = StageParamNames.stage1OperationName


  /**
   * Abstract function that gets the map of the output column names
   *
   * @param f1 feature 1
   * @param f2 feature 2
   * @return Map (name of param, value of param) of output column names
   */
  protected def getOutputsColNamesMap(f1: TransientFeature, f2: TransientFeature): Map[String, String]


  /**
   * Abstract functions that gets all the models with the helpers
   *
   * @return sequence of ModelInfo
   */
  protected def getModelInfo: Seq[ModelInfo[E]]


  // Map (name of param, value of param) of output column names
  lazy val outputsColNamesMap: Map[String, String] = getOutputsColNamesMap(in1, in2)
  lazy val labelColName: String = in1.name

  /**
   * Splits the data into training test and test set, balances the training set and selects the best model
   * Tests the model on the test set and prints results
   *
   * @param dataset
   * @return best model
   */
  final override def fit(dataset: Dataset[_]): SelectedModel = {
    import dataset.sparkSession.implicits._

    setInputSchema(dataset.schema).transformSchema(dataset.schema)

    val datasetWithID =
      if (dataset.columns.contains(DataFrameFieldNames.KeyFieldName)) {
        dataset.select(in1.name, in2.name, DataFrameFieldNames.KeyFieldName)
          .as[LabelFeaturesKey].persist()
      } else {
        dataset.select(in1.name, in2.name)
          .withColumn(ModelSelectorBaseNames.idColName, monotonically_increasing_id())
          .as[LabelFeaturesKey].persist()
      }

    val ModelData(trainData, met) = splitter match {
      case Some(spltr) => spltr.prepare(datasetWithID)
      case None => new ModelData(datasetWithID, new MetadataBuilder())
    }

    val bestModel = validator.validate(getModelInfo.filter(m => $(m.useModel)), trainData, in1.name, in2.name)
    bestModel.metadata.foreach(meta => setMetadata(meta.build))
    val bestClassifier = bestModel.model.parent
    log.info(s"Selected model : ${bestClassifier.getClass.getSimpleName}")
    log.info(s"With parameters : ${bestClassifier.extractParamMap()}")

    // set input and output params
    outputsColNamesMap.foreach { case (pname, pvalue) => bestModel.model.set(bestModel.model.getParam(pname), pvalue) }

    val builder = new MetadataBuilder().withMetadata(getMetadata()) // get cross val metrics
    builder.putString(ModelSelectorBaseNames.BestModelUid, bestModel.model.uid) // log bestModel uid (ie model type)
    builder.putString(ModelSelectorBaseNames.BestModelName, bestModel.name) // log bestModel name
    splitter.collect {
      case _: DataBalancer => builder.putMetadata(ModelSelectorBaseNames.ResampleValues, met)
      case _: DataCutter => builder.putMetadata(ModelSelectorBaseNames.CuttValues, met)
    }


    // add eval results to metadata
    val transformed = bestModel.model.transform(trainData)
    builder.putMetadata(ModelSelectorBaseNames.TrainingEval, evaluate(transformed).toMetadata)
    val allMetadata = builder.build().toSummaryMetadata()
    setMetadata(allMetadata)

    new SelectedModel(bestModel.model.asInstanceOf[Model[_ <: Model[_]]], outputsColNamesMap, uid)
      .setParent(this)
      .setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[OPVector])
      .setMetadata(allMetadata)
      .setEvaluators(evaluators)
  }

}

/**
 * Wrapper for the model returned by ModelSelector
 *
 * @param sparkMlStageIn best model
 * @param outputsColNamesMap column names
 * @param uid
 */
final class SelectedModel private[op]
(
  private val sparkMlStageIn: Model[_ <: Model[_]],
  val outputsColNamesMap: Map[String, String],
  val uid: String = UID[SelectedModel]
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val ttov: TypeTag[RealNN#Value]
) extends Model[SelectedModel]
  with OpPipelineStage2[RealNN, OPVector, RealNN]
  with SparkWrapperParams[Transformer with Params] with HasTestEval {

  setSparkMlStage(Option(sparkMlStageIn))

  val tto: TypeTag[RealNN] = tti1

  val operationName: String = StageParamNames.stage1OperationName
  lazy val labelColName: String = in1.name

  // TODO this is lost on serialization if we want to use the eval method here in eval runs as well as training
  // need to pass evaluators from origin stage to deserialized in OpPipelineStageReader.loadModel
  @transient private var evaluatorList: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty
  def setEvaluators(ev: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]]): this.type = {
    evaluatorList = ev
    this
  }
  override def evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = evaluatorList

  override def transform(dataset: Dataset[_]): DataFrame = {
    setInputSchema(dataset.schema)
    getSparkMlStage().get.transform(dataset)
  }
}

