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

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.evaluators._
import com.salesforce.op.stages.impl.ModelsToTry
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry._
import com.salesforce.op.stages.impl.selector.DefaultSelectorParams._
import com.salesforce.op.stages.impl.selector.ModelSelector
import com.salesforce.op.stages.impl.tuning._
import enumeratum.Enum
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames.{ModelType, EstimatorType}


/**
 * A factory for Regression Model Selector
 */
case object RegressionModelSelector {

  /**
   * Creates a new Regression Model Selector with a Cross Validation
   */
  def apply(): ModelSelector[ModelType, EstimatorType] = withCrossValidation()

  /**
   * Creates a new Regression Model Selector with a Cross Validation
   *
   * @param dataSplitter        instance that will split the data into training set and test set
   * @param numFolds            number of folds for cross validation (>= 2)
   * @param validationMetric    metric name in evaluation: RMSE, R2 etc
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is empty
   *                            and default evaluator is added to this list (here Evaluators.Regression)
   * @param seed                random seed
   * @param parallelism         level of parallelism used to schedule a number of models to be trained/evaluated
   *                            so that the jobs can be run concurrently
   * @return Regression Model Selector with a Cross Validation
   */
  def withCrossValidation(
    dataSplitter: Option[DataSplitter] = Option(DataSplitter()),
    numFolds: Int = ValidatorParamDefaults.NumFolds,
    validationMetric: OpRegressionEvaluatorBase[_] = Evaluators.Regression.rmse(),
    trainTestEvaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty,
    seed: Long = ValidatorParamDefaults.Seed,
    parallelism: Int = ValidatorParamDefaults.Parallelism
  ): ModelSelector[ModelType, EstimatorType] = {
    val cv = new OpCrossValidation[ModelType, EstimatorType](
      numFolds = numFolds, seed = seed, validationMetric, parallelism = parallelism
    )
    selector(
      cv, dataSplitter = dataSplitter, trainTestEvaluators = Seq(new OpRegressionEvaluator) ++ trainTestEvaluators
    )
  }

  /**
   * Creates a new Regression Model Selector with a Train Validation Split
   *
   * @param dataSplitter        instance that will split the data into training set and test set
   * @param trainRatio          ratio between training set and validation set (>= 0 && <= 1)
   * @param validationMetric    metric name in evaluation: RMSE, R2 etc
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is empty
   *                            and default evaluator is added to this list (here Evaluators.Regression)
   * @param seed                random seed
   * @param parallelism         level of parallelism used to schedule a number of models to be trained/evaluated
   *                            so that the jobs can be run concurrently
   * @return Regression Model Selector with a Train Validation Split
   */
  def withTrainValidationSplit(
    dataSplitter: Option[DataSplitter] = Option(DataSplitter()),
    trainRatio: Double = ValidatorParamDefaults.TrainRatio,
    validationMetric: OpRegressionEvaluatorBase[_] = Evaluators.Regression.rmse(),
    trainTestEvaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty,
    seed: Long = ValidatorParamDefaults.Seed,
    parallelism: Int = ValidatorParamDefaults.Parallelism
  ): ModelSelector[ModelType, EstimatorType] = {
    val ts = new OpTrainValidationSplit[ModelType, EstimatorType](
      trainRatio = trainRatio, seed = seed, validationMetric, parallelism = parallelism
    )
    selector(
      ts, dataSplitter = dataSplitter, trainTestEvaluators = Seq(new OpRegressionEvaluator) ++ trainTestEvaluators
    )
  }

  private def selector(
    validator: OpValidator[ModelType, EstimatorType],
    dataSplitter: Option[DataSplitter],
    trainTestEvaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]]
  ): ModelSelector[ModelType, EstimatorType] = {
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
 * Enumeration of possible regression models in Model Selector
 */
sealed trait RegressionModelsToTry extends ModelsToTry

object RegressionModelsToTry extends Enum[RegressionModelsToTry] {
  val values = findValues
  case object LinearRegression extends RegressionModelsToTry
  case object DecisionTreeRegression extends RegressionModelsToTry
  case object RandomForestRegression extends RegressionModelsToTry
  case object GBTRegression extends RegressionModelsToTry
  case object GeneralizedLinearRegression extends RegressionModelsToTry
}