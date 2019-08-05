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

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.evaluators._
import com.salesforce.op.stages.impl.ModelsToTry
import com.salesforce.op.stages.impl.classification.{BinaryClassificationModelsToTry => MTT}
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.{EstimatorType, ModelType}
import com.salesforce.op.stages.impl.selector.{DefaultSelectorParams, ModelSelector, ModelSelectorFactory}
import com.salesforce.op.stages.impl.tuning._
import enumeratum.Enum
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

import scala.concurrent.duration.Duration


/**
 * A factory for Binary Classification Model Selector
 */
case object BinaryClassificationModelSelector extends ModelSelectorFactory {

  /**
   * Default model types and model parameters for problem type
   */
  case object Defaults extends ModelDefaults[BinaryClassificationModelsToTry] {

    /**
     * Subset of models to use in model selector
     *
     * Note: [[OpNaiveBayes]], [[OpDecisionTreeClassifier]] and [[OpXGBoostClassifier]] are off by default
     */
    val modelTypesToUse: Seq[BinaryClassificationModelsToTry] = Seq(
      MTT.OpLogisticRegression, MTT.OpRandomForestClassifier, MTT.OpGBTClassifier, MTT.OpLinearSVC
    )

    /**
     * Default models and parameters (must be a def) to use in model selector
     *
     * @return defaults for problem type
     */
    def modelsAndParams: Seq[(EstimatorType, ParamGridBuilder)] = {
      val lr = new OpLogisticRegression()
      val lrParams = new ParamGridBuilder()
        .addGrid(lr.fitIntercept, DefaultSelectorParams.FitIntercept)
        .addGrid(lr.elasticNetParam, DefaultSelectorParams.ElasticNet)
        .addGrid(lr.maxIter, DefaultSelectorParams.MaxIterLin)
        .addGrid(lr.regParam, DefaultSelectorParams.Regularization)
        .addGrid(lr.standardization, DefaultSelectorParams.Standardized)
        .addGrid(lr.tol, DefaultSelectorParams.Tol)

      val rf = new OpRandomForestClassifier()
      val rfParams = new ParamGridBuilder()
        .addGrid(rf.maxDepth, DefaultSelectorParams.MaxDepth)
        .addGrid(rf.impurity, DefaultSelectorParams.ImpurityClass)
        .addGrid(rf.maxBins, DefaultSelectorParams.MaxBin)
        .addGrid(rf.minInfoGain, DefaultSelectorParams.MinInfoGain)
        .addGrid(rf.minInstancesPerNode, DefaultSelectorParams.MinInstancesPerNode)
        .addGrid(rf.numTrees, DefaultSelectorParams.MaxTrees)
        .addGrid(rf.subsamplingRate, DefaultSelectorParams.SubsampleRate)

      val gbt = new OpGBTClassifier()
      val gbtParams = new ParamGridBuilder()
        .addGrid(gbt.maxDepth, DefaultSelectorParams.MaxDepth)
        .addGrid(gbt.impurity, DefaultSelectorParams.ImpurityClass)
        .addGrid(gbt.maxBins, DefaultSelectorParams.MaxBin)
        .addGrid(gbt.minInfoGain, DefaultSelectorParams.MinInfoGain)
        .addGrid(gbt.minInstancesPerNode, DefaultSelectorParams.MinInstancesPerNode)
        .addGrid(gbt.maxIter, DefaultSelectorParams.MaxIterTree)
        .addGrid(gbt.subsamplingRate, DefaultSelectorParams.SubsampleRate)
        .addGrid(gbt.stepSize, DefaultSelectorParams.StepSize)

      val svc = new OpLinearSVC()
      val svcParams = new ParamGridBuilder()
        .addGrid(svc.regParam, DefaultSelectorParams.Regularization)
        .addGrid(svc.maxIter, DefaultSelectorParams.MaxIterLin)
        .addGrid(svc.fitIntercept, DefaultSelectorParams.FitIntercept)
        .addGrid(svc.tol, DefaultSelectorParams.Tol)
        .addGrid(svc.standardization, DefaultSelectorParams.Standardized)

      val nb = new OpNaiveBayes()
      val nbParams = new ParamGridBuilder()
        .addGrid(nb.smoothing, DefaultSelectorParams.NbSmoothing)

      val dt = new OpDecisionTreeClassifier()
      val dtParams = new ParamGridBuilder()
        .addGrid(dt.maxDepth, DefaultSelectorParams.MaxDepth)
        .addGrid(dt.impurity, DefaultSelectorParams.ImpurityClass)
        .addGrid(dt.maxBins, DefaultSelectorParams.MaxBin)
        .addGrid(dt.minInfoGain, DefaultSelectorParams.MinInfoGain)
        .addGrid(dt.minInstancesPerNode, DefaultSelectorParams.MinInstancesPerNode)

      val xgb = new OpXGBoostClassifier()
      val xgbParams = new ParamGridBuilder()
        .addGrid(xgb.numRound, DefaultSelectorParams.NumRound)
        .addGrid(xgb.eta, DefaultSelectorParams.Eta)
        .addGrid(xgb.maxDepth, DefaultSelectorParams.MaxDepth)
        .addGrid(xgb.minChildWeight, DefaultSelectorParams.MinChildWeight)

      Seq(lr -> lrParams, rf -> rfParams, gbt -> gbtParams, svc -> svcParams,
        nb -> nbParams, dt -> dtParams, xgb -> xgbParams)
    }
  }

  /**
   * Creates a new Binary Classification Model Selector with a Cross Validation
   */
  def apply(): ModelSelector[ModelType, EstimatorType] = withCrossValidation()

  /**
   * Creates a new Binary Classification Model Selector with a Cross Validation
   *
   * @param splitter            instance that will balance and split the data
   * @param numFolds            number of folds for cross validation (>= 2)
   * @param validationMetric    metric name in evaluation: AuROC or AuPR
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is empty
   *                            and default evaluator is added to this list (here Evaluators.BinaryClassification)
   * @param seed                random seed
   * @param stratify            whether or not stratify cross validation. Caution : setting that param to true might
   *                            impact the runtime
   * @param parallelism         level of parallelism used to schedule a number of models to be trained/evaluated
   *                            so that the jobs can be run concurrently
   * @param modelTypesToUse     list of model types to run grid search on must from supported types in
   *                            BinaryClassificationModelsToTry (OpLogisticRegression, OpRandomForestClassifier,
   *                            OpGBTClassifier, OpLinearSVC, OpDecisionTreeClassifier, OpNaiveBayes)
   * @param modelsAndParameters pass in an explicit list pairs of estimators and the accompanying hyperparameters to
   *                            for model selection Seq[(EstimatorType, Array[ParamMap])] where Estimator type must be
   *                            an Estimator that takes in a label (RealNN) and features (OPVector) and returns a
   *                            prediction (Prediction)
   * @param maxWait             maximum allowable time to wait for a model to finish running (default is 1 day)
   * @return Classification Model Selector with a Cross Validation
   */
  def withCrossValidation(
    splitter: Option[Splitter] = Option(DataSplitter()),
    numFolds: Int = ValidatorParamDefaults.NumFolds,
    validationMetric: OpBinaryClassificationEvaluatorBase[_] = Evaluators.BinaryClassification.auPR(),
    trainTestEvaluators: Seq[OpBinaryClassificationEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty,
    seed: Long = ValidatorParamDefaults.Seed,
    stratify: Boolean = ValidatorParamDefaults.Stratify,
    parallelism: Int = ValidatorParamDefaults.Parallelism,
    modelTypesToUse: Seq[BinaryClassificationModelsToTry] = Defaults.modelTypesToUse,
    modelsAndParameters: Seq[(EstimatorType, Array[ParamMap])] = Seq.empty,
    maxWait: Duration = ValidatorParamDefaults.MaxWait
  ): ModelSelector[ModelType, EstimatorType] = {
    val cv = new OpCrossValidation[ModelType, EstimatorType](
      numFolds = numFolds, seed = seed, evaluator = validationMetric, stratify = stratify,
      parallelism = parallelism, maxWait = maxWait
    )
    selector(cv,
      splitter = splitter,
      trainTestEvaluators = Seq(new OpBinaryClassificationEvaluator, new OpBinScoreEvaluator) ++ trainTestEvaluators,
      modelTypesToUse = modelTypesToUse,
      modelsAndParameters = modelsAndParameters,
      modelDefaults = Defaults
    )
  }

  /**
   * Creates a new Binary Classification Model Selector with a Train Validation Split
   *
   * @param splitter            instance that will balance and split the data
   * @param trainRatio          ratio between training set and validation set (>= 0 && <= 1)
   * @param validationMetric    metric name in evaluation: AuROC or AuPR
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is empty
   *                            and default evaluator is added to this list (here Evaluators.BinaryClassification)
   * @param seed                random seed
   * @param stratify            whether or not stratify train validation split.
   *                            Caution : setting that param to true might impact the runtime
   * @param parallelism         level of parallelism used to schedule a number of models to be trained/evaluated
   *                            so that the jobs can be run concurrently
   * @param modelTypesToUse     list of model types to run grid search on must from supported types in
   *                            BinaryClassificationModelsToTry (OpLogisticRegression, OpRandomForestClassifier,
   *                            OpGBTClassifier, OpLinearSVC, OpDecisionTreeClassifier, OpNaiveBayes)
   * @param modelsAndParameters pass in an explicit list pairs of estimators and the accompanying hyperparameters to
   *                            for model selection Seq[(EstimatorType, Array[ParamMap])] where Estimator type must be
   *                            an Estimator that takes in a label (RealNN) and features (OPVector) and returns a
   *                            prediction (Prediction)
   * @param maxWait             maximum allowable time to wait for a model to finish running (default is 1 day)
   * @return Classification Model Selector with a Train Validation Split
   */
  def withTrainValidationSplit(
    splitter: Option[Splitter] = Option(DataSplitter()),
    trainRatio: Double = ValidatorParamDefaults.TrainRatio,
    validationMetric: OpBinaryClassificationEvaluatorBase[_] = Evaluators.BinaryClassification.auPR(),
    trainTestEvaluators: Seq[OpBinaryClassificationEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty,
    seed: Long = ValidatorParamDefaults.Seed,
    stratify: Boolean = ValidatorParamDefaults.Stratify,
    parallelism: Int = ValidatorParamDefaults.Parallelism,
    modelTypesToUse: Seq[BinaryClassificationModelsToTry] = Defaults.modelTypesToUse,
    modelsAndParameters: Seq[(EstimatorType, Array[ParamMap])] = Seq.empty,
    maxWait: Duration = ValidatorParamDefaults.MaxWait
  ): ModelSelector[ModelType, EstimatorType] = {
    val ts = new OpTrainValidationSplit[ModelType, EstimatorType](
      trainRatio = trainRatio, seed = seed, validationMetric, stratify = stratify, parallelism = parallelism,
      maxWait = maxWait
    )
    selector(ts,
      splitter = splitter,
      trainTestEvaluators = Seq(new OpBinaryClassificationEvaluator) ++ trainTestEvaluators,
      modelTypesToUse = modelTypesToUse,
      modelsAndParameters = modelsAndParameters,
      modelDefaults = Defaults
    )
  }

}

/**
 * Enumeration of possible classification models in Model Selector
 */
sealed trait BinaryClassificationModelsToTry extends ModelsToTry

object BinaryClassificationModelsToTry extends Enum[BinaryClassificationModelsToTry] {
  val values = findValues
  case object OpLogisticRegression extends BinaryClassificationModelsToTry
  case object OpRandomForestClassifier extends BinaryClassificationModelsToTry
  case object OpGBTClassifier extends BinaryClassificationModelsToTry
  case object OpLinearSVC extends BinaryClassificationModelsToTry
  case object OpDecisionTreeClassifier extends BinaryClassificationModelsToTry
  case object OpNaiveBayes extends BinaryClassificationModelsToTry
  case object OpXGBoostClassifier extends BinaryClassificationModelsToTry
  case class Custom(private val modeType: Class[_ <: EstimatorType]) extends BinaryClassificationModelsToTry {
    override val entryName: String = modeType.getSimpleName
  }
}
