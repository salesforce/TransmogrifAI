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
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry._
import com.salesforce.op.stages.impl.selector.DefaultSelectorParams._
import com.salesforce.op.stages.impl.selector.ModelSelector
import com.salesforce.op.stages.impl.tuning._
import enumeratum.Enum
import org.apache.spark.ml.param.ParamMap
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames.{EstimatorType, ModelType}
import org.apache.spark.ml.tuning.ParamGridBuilder


/**
 * A factory for Binary Classification Model Selector
 */
case object BinaryClassificationModelSelector {

  private val modelNames: Seq[BinaryClassificationModelsToTry] = Seq(OpLogisticRegression, OpRandomForestClassifier,
    OpGBTClassifier, OpLinearSVC) // OpNaiveBayes and OpDecisionTreeClassifier off by default

  private val defaultModelsAndParams: Seq[(EstimatorType, Array[ParamMap])] = {
    val lr = new OpLogisticRegression()
    val lrParams = new ParamGridBuilder()

    val rf = new OpRandomForestClassifier()
    val rfParams = new ParamGridBuilder()

    val gbt = new OpGBTClassifier()
    val gbtParams = new ParamGridBuilder()

    val svc = new OpLinearSVC()
    val svcParams = new ParamGridBuilder()

    val nb = new OpRandomForestClassifier()
    val nbParams = new ParamGridBuilder()

    val dt = new OpRandomForestClassifier()
    val dtParams = new ParamGridBuilder()

    Seq(lr -> lrParams, rf -> rfParams, gbt -> gbtParams, svc -> svcParams, nb -> nbParams, dt -> dtParams)
      .asInstanceOf[Seq[(EstimatorType, Array[ParamMap])]]
  }

//  // Random forest defaults
//  .setRandomForestMaxDepth(MaxDepth: _*)
//    .setRandomForestImpurity(ImpurityClass)
//    .setRandomForestMaxBins(MaxBin)
//    .setRandomForestMinInfoGain(MinInfoGain: _*)
//    .setRandomForestMinInstancesPerNode(MinInstancesPerNode: _*)
//    .setRandomForestNumTrees(MaxTrees)
//    .setRandomForestSubsamplingRate(SubsampleRate)
//    // Logistic regression defaults
//    .setLogisticRegressionElasticNetParam(ElasticNet)
//    .setLogisticRegressionFitIntercept(FitIntercept)
//    .setLogisticRegressionMaxIter(MaxIterLin)
//    .setLogisticRegressionRegParam(Regularization: _*)
//    .setLogisticRegressionStandardization(Standardized)
//    .setLogisticRegressionTol(Tol)
//    // NB defaults
//    .setNaiveBayesModelType(NbModel)
//    .setNaiveBayesSmoothing(NbSmoothing)
//    // DT defaults
//    .setDecisionTreeImpurity(ImpurityClass)
//    .setDecisionTreeMaxBins(MaxBin)
//    .setDecisionTreeMaxDepth(MaxDepth: _*)
//    .setDecisionTreeMinInfoGain(MinInfoGain: _*)
//    .setDecisionTreeMinInstancesPerNode(MinInstancesPerNode: _*)
//

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
    modelTypesToUse: Seq[BinaryClassificationModelsToTry] = modelNames,
    modelsAndParameters: Seq[(EstimatorType, Array[ParamMap])] = defaultModelsAndParams
  ): ModelSelector[ModelType, EstimatorType] = {
    val cv = new OpCrossValidation[ModelType, EstimatorType](
      numFolds = numFolds, seed = seed, validationMetric, stratify = stratify, parallelism = parallelism
    )
    selector(
      cv, splitter = splitter, trainTestEvaluators = Seq(new OpBinaryClassificationEvaluator) ++ trainTestEvaluators,
      modelTypesToUse = modelTypesToUse, modelsAndParameters = modelsAndParameters
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
    modelTypesToUse: Seq[BinaryClassificationModelsToTry] = modelNames,
    modelsAndParameters: Seq[(EstimatorType, Array[ParamMap])] = defaultModelsAndParams
  ): ModelSelector[ModelType, EstimatorType] = {
    val ts = new OpTrainValidationSplit[ModelType, EstimatorType](
      trainRatio = trainRatio, seed = seed, validationMetric, stratify = stratify, parallelism = parallelism
    )
    selector(
      ts, splitter = splitter, trainTestEvaluators = Seq(new OpBinaryClassificationEvaluator) ++ trainTestEvaluators,
      modelTypesToUse = modelTypesToUse, modelsAndParameters = modelsAndParameters
    )
  }

  private def selector(
    validator: OpValidator[ModelType, EstimatorType],
    splitter: Option[Splitter],
    trainTestEvaluators: Seq[OpBinaryClassificationEvaluatorBase[_ <: EvaluationMetrics]],
    modelTypesToUse: Seq[BinaryClassificationModelsToTry],
    modelsAndParameters: Seq[(EstimatorType, Array[ParamMap])]
  ): ModelSelector[ModelType, EstimatorType] = {
    val modelStrings = modelTypesToUse.map(_.entryName)
    val modelsToUse = if (modelsAndParameters == defaultModelsAndParams) {
      modelsAndParameters.filter{ case (e, p) => modelStrings.contains(e.getClass.getSimpleName) }
    } else modelsAndParameters
    new ModelSelector(
      validator = validator,
      splitter = splitter,
      models = modelsToUse,
      evaluators = trainTestEvaluators
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
  case object OpLinearSVC extends BinaryClassificationModelsToTry // TODO need to ensure that eval works with this
  case object OpDecisionTreeClassifier extends BinaryClassificationModelsToTry
  case object OpNaiveBayes extends BinaryClassificationModelsToTry
}
