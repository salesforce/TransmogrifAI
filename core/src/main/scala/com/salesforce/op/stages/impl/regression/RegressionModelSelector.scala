/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.UID
import com.salesforce.op.evaluators._
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.stages.impl.regression.RegressorType.Regressor
import com.salesforce.op.stages.impl.selector.{ModelInfo, ModelSelectorBase}
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.stages.makeOutputName
import org.apache.spark.ml.Model
import org.apache.spark.sql.Dataset


/**
 * A factory for Regression Model Selector
 */
case object RegressionModelSelector {

  /**
   * Creates a new Regression Model Selector with a Cross Validation
   */
  def apply(): RegressionModelSelector = withCrossValidation()

  /**
   * Creates a new Regression Model Selector with a Cross Validation
   *
   * @param dataSplitter      instance that will split the data into training set and test set
   * @param numFolds          number of folds for cross validation (>= 2)
   * @param validationMetric  metric name in evaluation: RMSE, R2 etc
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is empty
   *                          and Ddefault evaluator is added to this list (here Evaluators.Regression)
   * @param seed              random seed
   * @return Regression Model Selector with a Cross Validation
   */
  def withCrossValidation(
    dataSplitter: Option[DataSplitter] = Option(DataSplitter()),
    numFolds: Int = ValidatorParamDefaults.NumFolds,
    validationMetric: OpRegressionEvaluatorBase[_] = Evaluators.Regression.rmse(),
    trainTestEvaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty,
    seed: Long = ValidatorParamDefaults.Seed
  ): RegressionModelSelector = {
    new RegressionModelSelector(
      validator = new OpCrossValidation[Regressor](numFolds, seed, validationMetric),
      dataSplitter = dataSplitter,
      trainTestEvaluators = Seq(new OpRegressionEvaluator) ++ trainTestEvaluators
    )
  }

  /**
   * Creates a new Regression Model Selector with a Train Validation Split
   *
   * @param dataSplitter      instance that will split the data into training set and test set
   * @param trainRatio        ratio between training set and validation set (>= 0 && <= 1)
   * @param validationMetric  metric name in evaluation: RMSE, R2 etc
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is empty
   *                          and Ddefault evaluator is added to this list (here Evaluators.Regression)
   * @param seed              random seed
   * @return Regression Model Selector with a Train Validation Split
   */
  def withTrainValidationSplit(
    dataSplitter: Option[DataSplitter] = Option(DataSplitter()),
    trainRatio: Double = ValidatorParamDefaults.TrainRatio,
    validationMetric: OpRegressionEvaluatorBase[_] = Evaluators.Regression.rmse(),
    trainTestEvaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty,
    seed: Long = ValidatorParamDefaults.Seed
  ): RegressionModelSelector = {
    new RegressionModelSelector(
      validator = new OpTrainValidationSplit[Regressor](
        trainRatio,
        seed,
        validationMetric
      ),
      dataSplitter = dataSplitter,
      trainTestEvaluators = Seq(new OpRegressionEvaluator) ++ trainTestEvaluators
    )
  }

}

/**
 * Model Selector that selects best regression model
 *
 * @param validator         validator used for the model selector
 * @param uid
 * @param dataSplitter      instance that will split the data into training set and test set
 * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation.
 *
 */
private[op] class RegressionModelSelector
(
  validator: OpValidator[Regressor],
  val dataSplitter: Option[DataSplitter],
  trainTestEvaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]],
  uid: String = UID[RegressionModelSelector]
) extends ModelSelectorBase[Regressor](validator = validator, splitter = dataSplitter,
  trainTestEvaluators = trainTestEvaluators, uid = uid)
  with SelectorRegressors {

  final override protected def evaluate(
    data: Dataset[_],
    labelColName: String,
    predictionColName: String,
    best: => Model[_ <: Model[_]]
  ): EvaluationMetrics = {
    val metricsMap = trainTestEvaluators.map { evaluator =>
      evaluator.name -> evaluator
        .setLabelCol(labelColName)
        .setPredictionCol(predictionColName)
        .evaluateAll(data)
    }.toMap

    MultiMetrics(metricsMap)
  }

  final override protected def getOutputsColNamesMap(f1: TransientFeature, f2: TransientFeature): Map[String, String] =
    Map(outputParam1Name -> makeOutputName(outputFeatureUid, Seq(f1, f2)))

  final override protected def getModelInfo: Seq[ModelInfo[Regressor]] = modelInfo
}
