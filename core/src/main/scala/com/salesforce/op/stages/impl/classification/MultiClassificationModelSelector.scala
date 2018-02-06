/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.UID
import com.salesforce.op.evaluators._
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry.{LogisticRegression, RandomForest}
import com.salesforce.op.stages.impl.classification.ProbabilisticClassifierType.ProbClassifier
import com.salesforce.op.stages.impl.selector.DefaultSelectorParams._
import com.salesforce.op.stages.impl.selector.StageOperationName
import com.salesforce.op.stages.impl.tuning._
import com.salesforce.op.stages.sparkwrappers.generic.{SwQuaternaryTransformer, SwTernaryTransformer}
import org.apache.spark.ml.Model
import org.apache.spark.sql.Dataset


/**
 * A factory for Multi Classification Model Selector
 */
case object MultiClassificationModelSelector {

  /**
   * Creates a new Multi Classification Model Selector with a Cross Validation
   */
  def apply(): MultiClassificationModelSelector = withCrossValidation()

  /**
   * Creates a new Multi Classification Model Selector with a Cross Validation
   *
   * @param splitter         instance that will split the data
   * @param numFolds         number of folds for cross validation (>= 2)
   * @param validationMetric metric name in evaluation: Accuracy, Precision, Recall or F1
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is empty
   *                          and default evaluator is added to this list (here OpMultiClassificationEvaluator)
   * @param seed             random seed
   * @return MultiClassification Model Selector with a Cross Validation
   */
  def withCrossValidation(
    splitter: Option[DataCutter] = Option(DataCutter()),
    numFolds: Int = ValidatorParamDefaults.NumFolds,
    validationMetric: OpMultiClassificationEvaluatorBase[_] = Evaluators.MultiClassification.error(),
    trainTestEvaluators: Seq[OpMultiClassificationEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty,
    seed: Long = ValidatorParamDefaults.Seed
  ): MultiClassificationModelSelector = {
    selector(
      new OpCrossValidation[ProbClassifier](numFolds, seed, validationMetric),
      splitter = splitter,
      trainTestEvaluators = Seq(new OpMultiClassificationEvaluator) ++ trainTestEvaluators
    )
  }

  /**
   * Creates a new Multi Classification Model Selector with a Train Validation Split
   *
   * @param splitter         instance that will split the data
   * @param trainRatio       ratio between training set and validation set (>= 0 && <= 1)
   * @param validationMetric metric name in evaluation: AuROC or AuPR
   * @param trainTestEvaluators List of evaluators applied on training + holdout data for evaluation. Default is empty
   *                          and default evaluator is added to this list (here OpMultiClassificationEvaluator)
   * @param seed             random seed
   * @return MultiClassification Model Selector with a Train Validation Split
   */
  def withTrainValidationSplit(
    splitter: Option[DataCutter] = Option(DataCutter()),
    trainRatio: Double = ValidatorParamDefaults.TrainRatio,
    validationMetric: OpMultiClassificationEvaluatorBase[_] = Evaluators.MultiClassification.error(),
    trainTestEvaluators: Seq[OpMultiClassificationEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty,
    seed: Long = ValidatorParamDefaults.Seed
  ): MultiClassificationModelSelector = {
    selector(
      new OpTrainValidationSplit[ProbClassifier](
        trainRatio,
        seed,
        validationMetric
      ),
      splitter = splitter,
      trainTestEvaluators = Seq(new OpMultiClassificationEvaluator) ++ trainTestEvaluators
    )
  }

  private def selector(
    validator: OpValidator[ProbClassifier],
    splitter: Option[DataCutter],
    trainTestEvaluators: Seq[OpMultiClassificationEvaluatorBase[_ <: EvaluationMetrics]]
  ): MultiClassificationModelSelector = {
    new MultiClassificationModelSelector(
      validator = validator,
      splitter = splitter,
      evaluators = trainTestEvaluators
    ) // models on by default
      .setModelsToTry(RandomForest, LogisticRegression)
      // Random forest defaults
      .setRandomForestMaxDepth(MaxDepth: _*)
      .setRandomForestImpurity(ImpurityClass)
      .setRandomForestMaxBins(MaxBin)
      .setRandomForestMinInfoGain(MinInfoGain: _*)
      .setRandomForestMinInstancesPerNode(MinInstancesPerNode: _*)
      .setRandomForestNumTrees(MaxTrees)
      .setRandomForestSubsamplingRate(SubsampleRate)
      // Logistic regression defaults
      .setLogisticRegressionElasticNetParam(ElasticNet)
      .setLogisticRegressionFitIntercept(FitIntercept)
      .setLogisticRegressionMaxIter(MaxIterLin)
      .setLogisticRegressionRegParam(Regularization: _*)
      .setLogisticRegressionStandardization(Standardized)
      .setLogisticRegressionTol(Tol)
      // NB defaults
      .setNaiveBayesModelType(NbModel)
      .setNaiveBayesSmoothing(NbSmoothing)
      // DT defaults
      .setDecisionTreeImpurity(ImpurityClass)
      .setDecisionTreeMaxBins(MaxBin)
      .setDecisionTreeMaxDepth(MaxDepth: _*)
      .setDecisionTreeMinInfoGain(MinInfoGain: _*)
      .setDecisionTreeMinInstancesPerNode(MinInstancesPerNode: _*)
  }
}


/**
 * Multi Classification Model Selector
 *
 * @param validator Cross Validation or Train Validation Split
 * @param splitter  instance that will split the data
 * @param evaluators List of evaluators applied on training + holdout data for evaluation.
 * @param uid
 */
private[op] class MultiClassificationModelSelector
(
  override val validator: OpValidator[ProbClassifier],
  override val splitter: Option[DataCutter],
  override val evaluators: Seq[OpMultiClassificationEvaluatorBase[_ <: EvaluationMetrics]],
  override val uid: String = UID[MultiClassificationModelSelector]
) extends ClassificationModelSelector(validator, splitter, evaluators, uid) with StageOperationName {

  override private[classification] val stage1uid: String = UID[Stage1BinaryClassificationModelSelector]

  lazy val stage1 = new Stage1MultiClassificationModelSelector(validator = validator,
    splitter = splitter.asInstanceOf[Option[DataCutter]], evaluators = evaluators,
    uid = stage1uid, stage2uid = stage2uid, stage3uid = stage3uid)
}



/**
 * Stage 1 of MultiClassificationModelSelector
 *
 * @param validator Cross Validation or Train Validation Split
 * @param splitter  instance that will split the data
 * @param evaluators List of evaluators applied on training + holdout data for evaluation.
 * @param uid
 */
private[op] class Stage1MultiClassificationModelSelector
(
  validator: OpValidator[ProbClassifier],
  splitter: Option[DataCutter],
  evaluators: Seq[OpMultiClassificationEvaluatorBase[_ <: EvaluationMetrics]],
  uid: String = UID[Stage1MultiClassificationModelSelector],
  stage2uid: String = UID[SwTernaryTransformer[_, _, _, _, _]],
  stage3uid: String = UID[SwQuaternaryTransformer[_, _, _, _, _, _]]
) extends Stage1ClassificationModelSelector(validator, splitter, evaluators, uid, stage2uid, stage3uid)
