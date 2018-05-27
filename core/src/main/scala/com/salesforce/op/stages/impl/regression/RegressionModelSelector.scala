/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.UID
import com.salesforce.op.evaluators._
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry._
import com.salesforce.op.stages.impl.regression.RegressorType._
import com.salesforce.op.stages.impl.selector.DefaultSelectorParams._
import com.salesforce.op.stages.impl.selector.{ModelSelectorBase, StageParamNames}
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.stages.makeOutputName

import scala.util.Try


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
