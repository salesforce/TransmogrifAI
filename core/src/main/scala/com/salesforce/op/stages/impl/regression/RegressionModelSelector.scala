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
import com.salesforce.op.stages.impl.regression.{RegressionModelsToTry => MTT}
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.{EstimatorType, ModelType}
import com.salesforce.op.stages.impl.selector.{DefaultSelectorParams, ModelSelector, ModelSelectorFactory}
import com.salesforce.op.stages.impl.tuning._
import enumeratum.Enum
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

import scala.concurrent.duration.Duration


/**
 * A factory for Regression Model Selector
 */
case object RegressionModelSelector extends ModelSelectorFactory {

  /**
   * Default model types and model parameters for problem type
   */
  case object Defaults extends ModelDefaults[RegressionModelsToTry] {

    /**
     * Subset of models to use in model selector
     *
     * Note: [[OpDecisionTreeRegressor]] and [[OpXGBoostRegressor]] are off by default
     */
    val modelTypesToUse: Seq[RegressionModelsToTry] = Seq(
      MTT.OpLinearRegression, MTT.OpRandomForestRegressor, MTT.OpGBTRegressor
    )

    /**
     * Default models and parameters (must be a def) to use in model selector
     *
     * @return defaults for problem type
     */
    def modelsAndParams: Seq[(EstimatorType, ParamGridBuilder)] = {
      val lr = new OpLinearRegression()
      val lrParams = new ParamGridBuilder()
        .addGrid(lr.fitIntercept, DefaultSelectorParams.FitIntercept)
        .addGrid(lr.elasticNetParam, DefaultSelectorParams.ElasticNet)
        .addGrid(lr.maxIter, DefaultSelectorParams.MaxIterLin)
        .addGrid(lr.regParam, DefaultSelectorParams.Regularization)
        .addGrid(lr.solver, DefaultSelectorParams.RegSolver)
        .addGrid(lr.standardization, DefaultSelectorParams.Standardized)
        .addGrid(lr.tol, DefaultSelectorParams.Tol)

      val rf = new OpRandomForestRegressor()
      val rfParams = new ParamGridBuilder()
        .addGrid(rf.maxDepth, DefaultSelectorParams.MaxDepth)
        .addGrid(rf.maxBins, DefaultSelectorParams.MaxBin)
        .addGrid(rf.minInfoGain, DefaultSelectorParams.MinInfoGain)
        .addGrid(rf.minInstancesPerNode, DefaultSelectorParams.MinInstancesPerNode)
        .addGrid(rf.numTrees, DefaultSelectorParams.MaxTrees)
        .addGrid(rf.subsamplingRate, DefaultSelectorParams.SubsampleRate)

      val gbt = new OpGBTRegressor()
      val gbtParams = new ParamGridBuilder()
        .addGrid(gbt.lossType, DefaultSelectorParams.TreeLossType)
        .addGrid(gbt.maxDepth, DefaultSelectorParams.MaxDepth)
        .addGrid(gbt.maxBins, DefaultSelectorParams.MaxBin)
        .addGrid(gbt.minInfoGain, DefaultSelectorParams.MinInfoGain)
        .addGrid(gbt.minInstancesPerNode, DefaultSelectorParams.MinInstancesPerNode)
        .addGrid(gbt.maxIter, DefaultSelectorParams.MaxIterTree)
        .addGrid(gbt.subsamplingRate, DefaultSelectorParams.SubsampleRate)
        .addGrid(gbt.stepSize, DefaultSelectorParams.StepSize)

      val dt = new OpDecisionTreeRegressor()
      val dtParams = new ParamGridBuilder()
        .addGrid(dt.maxDepth, DefaultSelectorParams.MaxDepth)
        .addGrid(dt.maxBins, DefaultSelectorParams.MaxBin)
        .addGrid(dt.minInfoGain, DefaultSelectorParams.MinInfoGain)
        .addGrid(dt.minInstancesPerNode, DefaultSelectorParams.MinInstancesPerNode)

      val glr = new OpGeneralizedLinearRegression()
      val glrParams = new ParamGridBuilder()
        .addGrid(glr.fitIntercept, DefaultSelectorParams.FitIntercept)
        .addGrid(glr.family, DefaultSelectorParams.DistFamily)
        .addGrid(glr.link, DefaultSelectorParams.LinkFunction)
        .addGrid(glr.maxIter, DefaultSelectorParams.MaxIterLin)
        .addGrid(glr.regParam, DefaultSelectorParams.Regularization)
        .addGrid(glr.tol, DefaultSelectorParams.Tol)

      val xgb = new OpXGBoostRegressor()
      val xgbParams = new ParamGridBuilder()
        .addGrid(xgb.numRound, DefaultSelectorParams.NumRound)
        .addGrid(xgb.eta, DefaultSelectorParams.Eta)
        .addGrid(xgb.maxDepth, DefaultSelectorParams.MaxDepth)
        .addGrid(xgb.minChildWeight, DefaultSelectorParams.MinChildWeight)

      Seq(lr -> lrParams, rf -> rfParams, gbt -> gbtParams, dt -> dtParams, glr -> glrParams, xgb -> xgbParams)
    }
  }

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
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is
   *                            the standard OpRegressionEvaluator.
   * @param seed                random seed
   * @param parallelism         level of parallelism used to schedule a number of models to be trained/evaluated
   *                            so that the jobs can be run concurrently
   * @param modelTypesToUse     list of model types to run grid search on must from supported types in
   *                            RegressionModelsToTry (OpLinearRegression, OpDecisionTreeRegressor,
   *                            OpRandomForestRegressor, OpGBTRegressor, OpGeneralizedLinearRegression)
   * @param modelsAndParameters pass in an explicit list pairs of estimators and the accompanying hyperparameters to
   *                            for model selection Seq[(EstimatorType, Array[ParamMap])] where Estimator type must be
   *                            an Estimator that takes in a label (RealNN) and features (OPVector) and returns a
   *                            prediction (Prediction)
   * @param maxWait             maximum allowable time to wait for a model to finish running (default is 1 day)
   * @return Regression Model Selector with a Cross Validation
   */
  def withCrossValidation(
    dataSplitter: Option[DataSplitter] = Option(DataSplitter()),
    numFolds: Int = ValidatorParamDefaults.NumFolds,
    validationMetric: OpRegressionEvaluatorBase[_ <: EvaluationMetrics] = Evaluators.Regression.rmse(),
    trainTestEvaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]] = Seq(new OpRegressionEvaluator),
    seed: Long = ValidatorParamDefaults.Seed,
    parallelism: Int = ValidatorParamDefaults.Parallelism,
    modelTypesToUse: Seq[RegressionModelsToTry] = Defaults.modelTypesToUse,
    modelsAndParameters: Seq[(EstimatorType, Array[ParamMap])] = Seq.empty,
    maxWait: Duration = ValidatorParamDefaults.MaxWait
  ): ModelSelector[ModelType, EstimatorType] = {
    val cv = new OpCrossValidation[ModelType, EstimatorType](
      numFolds = numFolds, seed = seed, evaluator = validationMetric, parallelism = parallelism, maxWait = maxWait
    )
    // For backwards compatibility, make sure evaluators always include the defaults
    val allEvaluators = addDefaultEvaluators(trainTestEvaluators, Seq(new OpRegressionEvaluator))

    selector(cv,
      splitter = dataSplitter,
      trainTestEvaluators = allEvaluators,
      modelTypesToUse = modelTypesToUse,
      modelsAndParameters = modelsAndParameters,
      modelDefaults = Defaults
    )
  }


  /**
   * Creates a new Regression Model Selector with a Train Validation Split
   *
   * @param dataSplitter        instance that will split the data into training set and test set
   * @param trainRatio          ratio between training set and validation set (>= 0 && <= 1)
   * @param validationMetric    metric name in evaluation: RMSE, R2 etc
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is
   *                            the standard OpRegressionEvaluator.
   * @param seed                random seed
   * @param parallelism         level of parallelism used to schedule a number of models to be trained/evaluated
   *                            so that the jobs can be run concurrently
   * @param modelTypesToUse     list of model types to run grid search on must from supported types in
   *                            RegressionModelsToTry (OpLinearRegression, OpDecisionTreeRegressor,
   *                            OpRandomForestRegressor, OpGBTRegressor, OpGeneralizedLinearRegression)
   * @param modelsAndParameters pass in an explicit list pairs of estimators and the accompanying hyperparameters to
   *                            for model selection Seq[(EstimatorType, Array[ParamMap])] where Estimator type must be
   *                            an Estimator that takes in a label (RealNN) and features (OPVector) and returns a
   *                            prediction (Prediction)
   * @param maxWait             maximum allowable time to wait for a model to finish running (default is 1 day)
   * @return Regression Model Selector with a Train Validation Split
   */
  def withTrainValidationSplit(
    dataSplitter: Option[DataSplitter] = Option(DataSplitter()),
    trainRatio: Double = ValidatorParamDefaults.TrainRatio,
    validationMetric: OpRegressionEvaluatorBase[_ <: EvaluationMetrics] = Evaluators.Regression.rmse(),
    trainTestEvaluators: Seq[OpRegressionEvaluatorBase[_ <: EvaluationMetrics]] = Seq(new OpRegressionEvaluator),
    seed: Long = ValidatorParamDefaults.Seed,
    parallelism: Int = ValidatorParamDefaults.Parallelism,
    modelTypesToUse: Seq[RegressionModelsToTry] = Defaults.modelTypesToUse,
    modelsAndParameters: Seq[(EstimatorType, Array[ParamMap])] = Seq.empty,
    maxWait: Duration = ValidatorParamDefaults.MaxWait
  ): ModelSelector[ModelType, EstimatorType] = {
    val ts = new OpTrainValidationSplit[ModelType, EstimatorType](
      trainRatio = trainRatio, seed = seed, validationMetric, parallelism = parallelism
    )
    // For backwards compatibility, make sure evaluators always include the defaults
    val allEvaluators = addDefaultEvaluators(trainTestEvaluators, Seq(new OpRegressionEvaluator))

    selector(ts,
      splitter = dataSplitter,
      trainTestEvaluators = allEvaluators,
      modelTypesToUse = modelTypesToUse,
      modelsAndParameters = modelsAndParameters,
      modelDefaults = Defaults
    )
  }

}

/**
 * Enumeration of possible regression models in Model Selector
 */
sealed trait RegressionModelsToTry extends ModelsToTry

object RegressionModelsToTry extends Enum[RegressionModelsToTry] {
  val values = findValues
  case object OpLinearRegression extends RegressionModelsToTry
  case object OpDecisionTreeRegressor extends RegressionModelsToTry
  case object OpRandomForestRegressor extends RegressionModelsToTry
  case object OpGBTRegressor extends RegressionModelsToTry
  case object OpGeneralizedLinearRegression extends RegressionModelsToTry
  case object OpXGBoostRegressor extends RegressionModelsToTry
  case class Custom(private val modeType: Class[_ <: EstimatorType]) extends RegressionModelsToTry {
    override val entryName: String = modeType.getSimpleName
  }
}
