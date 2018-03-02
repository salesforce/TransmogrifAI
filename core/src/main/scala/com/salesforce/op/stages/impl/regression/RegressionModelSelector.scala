/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.UID
import com.salesforce.op.evaluators._
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry._
import com.salesforce.op.stages.impl.regression.RegressorType._
import com.salesforce.op.stages.impl.selector.DefaultSelectorParams._
import com.salesforce.op.stages.impl.selector.{ModelInfo, ModelSelectorBase, StageParamNames}
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
   *                          and default evaluator is added to this list (here Evaluators.Regression)
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
    selector(
      validator = new OpCrossValidation[RegressorModel, Regressor](numFolds, seed, validationMetric),
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
   *                          and default evaluator is added to this list (here Evaluators.Regression)
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
    selector(
      validator = new OpTrainValidationSplit[RegressorModel, Regressor](
        trainRatio,
        seed,
        validationMetric
      ),
      dataSplitter = dataSplitter,
      trainTestEvaluators = Seq(new OpRegressionEvaluator) ++ trainTestEvaluators
    )
  }

  private def selector(
    validator: OpValidator[RegressorModel, Regressor],
    dataSplitter: Option[DataSplitter],
    trainTestEvaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]]): RegressionModelSelector = {
    new RegressionModelSelector(
      validator = validator,
      dataSplitter = dataSplitter,
      evaluators = trainTestEvaluators
    ) // models on by default
      .setModelsToTry(RandomForestRegression, LinearRegression, GBTRegression)
      // Random forest defaults
      .setRandomForestMaxDepth(MaxDepth: _*)
      .setRandomForestImpurity(ImpurityReg)
      .setRandomForestMaxBins(MaxBin)
      .setRandomForestMinInfoGain(MinInfoGain: _*)
      .setRandomForestMinInstancesPerNode(MinInstancesPerNode: _*)
      .setRandomForestNumTrees(MaxTrees)
      .setRandomForestSubsamplingRate(SubsampleRate)
      // Linear regression defaults
      .setLinearRegressionElasticNetParam(ElasticNet)
      .setLinearRegressionFitIntercept(FitIntercept)
      .setLinearRegressionMaxIter(MaxIterLin)
      .setLinearRegressionRegParam(Regularization: _*)
      .setLinearRegressionSolver(RegSolver)
      .setLinearRegressionStandardization(Standardized)
      .setLinearRegressionTol(Tol)
      // GBT defaults
      .setGradientBoostedTreeLossType(TreeLossType)
      .setGradientBoostedTreeMaxBins(MaxBin)
      .setGradientBoostedTreeMaxDepth(MaxDepth: _*)
      .setGradientBoostedTreeMinInfoGain(MinInfoGain: _*)
      .setGradientBoostedTreeMaxIter(MaxIterTree)
      .setGradientBoostedTreeMinInstancesPerNode(MinInstancesPerNode: _*)
      .setGradientBoostedTreeStepSize(StepSize)
      .setGradientBoostedTreeImpurity(ImpurityReg)
      .setGradientBoostedTreeSubsamplingRate(SubsampleRate)
      // DT defaults
      .setDecisionTreeImpurity(ImpurityReg)
      .setDecisionTreeMaxBins(MaxBin)
      .setDecisionTreeMaxDepth(MaxDepth: _*)
      .setDecisionTreeMinInfoGain(MinInfoGain: _*)
      .setDecisionTreeMinInstancesPerNode(MinInstancesPerNode: _*)
  }

}

/**
 * Model Selector that selects best regression model
 *
 * @param validator         validator used for the model selector
 * @param uid
 * @param dataSplitter      instance that will split the data into training set and test set
 * @param evaluators List of evaluators applied on training + holdout data for evaluation.
 *
 */
private[op] class RegressionModelSelector
(
  validator: OpValidator[RegressorModel, Regressor],
  val dataSplitter: Option[DataSplitter],
  evaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]],
  uid: String = UID[RegressionModelSelector]
) extends ModelSelectorBase[RegressorModel, Regressor](validator = validator, splitter = dataSplitter,
  evaluators = evaluators, uid = uid)
  with SelectorRegressors {

  final override protected def getOutputsColNamesMap(f1: TransientFeature, f2: TransientFeature): Map[String, String] =
    Map(StageParamNames.outputParam1Name -> makeOutputName(outputFeatureUid, Seq(f1, f2)))

}
