/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.UID
import com.salesforce.op.evaluators._
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.stages._
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.{Estimator, Model, PipelineStage, Transformer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.MetadataBuilder
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.reflect.runtime.universe._


case object ModelSelectorBaseNames {
  val TrainValSplitResults = "trainValidationSplitResults"
  val CrossValResults = "crossValidationResults"
  val TrainingEval = "trainingSetEvaluationResults"
  val HoldOutEval = "testSetEvaluationResults"
  val ResampleValues = "resamplingValues"
  val BestModelUid = "bestModelUID"
  val BestModelName = "bestModelName"
  val Positive = "positiveLabels"
  val Negative = "negativeLabels"
  val Desired = "desiredFraction"
  val UpSample = "upSamplingFraction"
  val DownSample = "downSamplingFraction"
  val idColName = "rowId"
}


/**
 * Factory to implement a Model Selector. Model Selector has only one output, it can be used for the stage 1 of a
 * model selector that outputs prediction, raw prediction and probability
 *
 * @param validator         validator used for the selection. It can be either CrossValidation or TrainValidationSplit
 * @param splitter          to split and/or balance the dataset
 * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation.
 * @param uid
 * @param tti1
 * @param tto
 * @tparam E Type parameter of the estimators used in the model selector
 */
private[op] abstract class ModelSelectorBase[E <: Estimator[_]]
(
  val validator: OpValidator[E],
  val splitter: Option[Splitter],
  val trainTestEvaluators: Seq[OpEvaluatorBase[_]],
  val uid: String = UID[ModelSelectorBase[E]]
)(
  implicit val tti1: TypeTag[OPVector],
  val tto: TypeTag[RealNN],
  val ttov: TypeTag[RealNN#Value]
) extends Estimator[SelectedModel]
  with OpPipelineStage2[RealNN, OPVector, RealNN]
  with Stage1ParamNamesBase {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  final override def operationName: String = stage1OperationName

  /**
   * Abstract function that gets the map of the output column names
   *
   * @param f1 feature 1
   * @param f2 feature 2
   * @return Map (name of param, value of param) of output column names
   */
  protected def getOutputsColNamesMap(f1: TransientFeature, f2: TransientFeature): Map[String, String]

  /**
   * Abstract function that evaluates the selected model on the test set
   *
   * @param data              data transformed by best model
   * @param labelColName      name of the labels column
   * @param predictionColName name of the prediction columns
   * @param best              best model
   * @return EvaluationMetrics
   */
  protected def evaluate(
    data: Dataset[_],
    labelColName: String,
    predictionColName: String,
    best: => Model[_ <: Model[_]]
  ): EvaluationMetrics


  /**
   * Abstract functions that gets all the models with the helpers
   *
   * @return sequence of ModelInfo
   */
  protected def getModelInfo: Seq[ModelInfo[E]]


  /**
   * Finds the best model
   *
   * @param data dataset
   * @return best model
   */
  private def findBestModel[M <: Model[M]](data: Dataset[_], hasIdColumn: Boolean): BestModel[M] = {
    // Remove Logging of OWLQN used in LogisticRegression
    Logger.getLogger("breeze.optimize.OWLQN").setLevel(Level.WARN)

    // Remove Logging of LBFGS used in LogisticRegression
    Logger.getLogger("breeze.optimize.LBFGS").setLevel(Level.WARN)

    val models: Seq[(E, Array[ParamMap])] = getModelInfo.filter(m => $(m.use)).map {
      case ModelInfo(e, g, b) => (e.asInstanceOf[E], g.build())
    }

    val fittedModels = models.collect {
      case (model, paramGrids) =>
        val pi1 = model.getParam(inputParam1Name)
        val pi2 = model.getParam(inputParam2Name)
        model.set(pi1, in1.name).set(pi2, in2.name)

        validator.validate(
          estimator = model.asInstanceOf[E],
          paramGrids = paramGrids,
          label = in1.name,
          hasIdColumn = hasIdColumn
        ).fit(data) -> paramGrids
    }

    // Put model info in summary metadata
    val allModelsStr = models.map(_._1.getClass.getSimpleName).mkString(", ")
    val bestModel = validator.bestModel[M](allModelsStr, fittedModels, new MetadataBuilder())
    bestModel.metadata.foreach(meta => setMetadata(meta.build))
    val bestClassifier = bestModel.model.parent
    log.info(s"Selected model : ${bestClassifier.getClass.getSimpleName}")
    log.info(s"With parameters : ${bestClassifier.extractParamMap()}")

    bestModel
  }


  /**
   * Get metrics on test set and put in metadata
   *
   * @param builder   metadata
   * @param bestModel model
   * @param trainData training data to evaluate
   * @param testData  optional test data to evaluate
   */
  private def putEvalResultsInMetadata(
    builder: MetadataBuilder,
    bestModel: Model[_ <: Model[_]],
    trainData: Dataset[_],
    testData: Option[Dataset[_]] = None
  ): Unit = {

    log.info("Evaluating the selected model on training set: \n")
    val trainMetrics = evaluateBestModel(bestModel, trainData)
    builder.putMetadata(ModelSelectorBaseNames.TrainingEval, trainMetrics.toMetadata)

    testData.map { case holdOutData =>
      log.info("Evaluating the selected model on hold out set: \n")
      val holdOutMetrics = evaluateBestModel(bestModel, holdOutData)
      builder.putMetadata(ModelSelectorBaseNames.HoldOutEval, holdOutMetrics.toMetadata)
    }
  }

  /**
   * Evaluate metrics of the best model on given data
   *
   * @param best model to use for evaluation
   * @param data data
   *
   * @return evaluationMetrics
   */
  private def evaluateBestModel(
    best: Model[_ <: Model[_]], data: Dataset[_]): EvaluationMetrics = {

    lazy val labelColName = in1.name
    val predictionColName = outputsColNamesMap(outputParam1Name)
    val transformedTest = best.transform(data)


    evaluate(transformedTest, labelColName, predictionColName, best)
  }

  // Map (name of param, value of param) of output column names
  lazy val outputsColNamesMap: Map[String, String] = getOutputsColNamesMap(in1, in2)


  /**
   * Splits the data into training test and test set, balances the training set and selects the best model
   * Tests the model on the test set and prints results
   *
   * @param dataset
   * @return best model
   */
  final override def fit(dataset: Dataset[_]): SelectedModel = {
    import dataset.sparkSession.implicits._

    val datasetWithID = dataset.select(in1.name, in2.name)
      .withColumn(ModelSelectorBaseNames.idColName, monotonically_increasing_id())
      .as[(Double, Vector, Double)].persist()

    val ModelSplitData(trainData, testData, met, hasLeak) = splitter match {
      case Some(spltr) => spltr.split(datasetWithID)
      case None => ModelSplitData(
        train = datasetWithID,
        test = datasetWithID,
        metadata = new MetadataBuilder().build(),
        hasLeakage = false
      )
    }

    val bestModel = findBestModel(trainData, hasLeak)

    // set input and output params
    outputsColNamesMap.foreach { case (pname, pvalue) => bestModel.model.set(bestModel.model.getParam(pname), pvalue) }

    val selectedModel =
      new SelectedModel(bestModel.model, uid = uid)
        .setParent(this)
        .setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[OPVector])

    val builder = new MetadataBuilder().withMetadata(getMetadata()) // get cross val metrics
    builder.putString(ModelSelectorBaseNames.BestModelUid, bestModel.model.uid) // log bestModel uid (ie model type)
    builder.putString(ModelSelectorBaseNames.BestModelName, bestModel.name) // log bestModel name

    splitter.collect {
      case d: DataBalancer =>
        builder.putMetadata(ModelSelectorBaseNames.ResampleValues, met)
        if (d.getSplitData) putEvalResultsInMetadata(
          builder = builder,
          bestModel = bestModel.model,
          trainData = trainData,
          testData = Option(testData)
        )
      case _: DataSplitter =>
        putEvalResultsInMetadata(
          builder = builder,
          bestModel = bestModel.model,
          trainData = trainData,
          testData = Option(testData)
        )
      case _ => putEvalResultsInMetadata(
        builder = builder,
        bestModel = bestModel.model,
        trainData = trainData
      )
    }
    val allMetadata = builder.build().toSummaryMetadata()
    setMetadata(allMetadata)
    selectedModel.setMetadata(allMetadata)
  }

}

/**
 * Wrapper for the model returned by ModelSelector
 *
 * @param sparkMlStageIn best model
 * @param uid
 */
final class SelectedModel
(
  private val sparkMlStageIn: Model[_ <: Model[_]],
  val uid: String = UID[SelectedModel]
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val ttov: TypeTag[RealNN#Value]
) extends Model[SelectedModel]
  with OpPipelineStage2[RealNN, OPVector, RealNN]
  with Stage1ParamNamesBase
  with SparkWrapperParams[Transformer with Params] {

  setSparkMlStage(Option(sparkMlStageIn))

  val tto: TypeTag[RealNN] = tti1

  override def operationName: String = stage1OperationName

  override def transform(dataset: Dataset[_]): DataFrame = getSparkMlStage().get.transform(dataset)
}

