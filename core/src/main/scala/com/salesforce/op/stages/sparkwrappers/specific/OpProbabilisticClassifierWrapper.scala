/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.sparkwrappers.generic.SwThreeStageBinaryEstimator
import org.apache.spark.ml.SparkMLSharedParamConstants
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap

/**
 * Wraps a spark ML probabilistic classifier.  In SparkML, a probabilistic classifier is anything that inherits
 * from [[ProbabilisticClassifier]].  Examples of these probabilistic classifiers
 * include: RandomForestClassifier, NaiveBayes, LogisticRegression, and DecisionTreeClassifier.
 * These classifiers in spark ML output not a single column, but 3: (1) the raw unnormalized scores for each class,
 * (2) the probabilistic classification (normalized raw scores), and
 * (3) the labels of the output (e.g. max unnormalized score).
 * The defining characteristic of classifiers intended to be wrapped by this class is that they output a model which
 * takes in 2 columns as input (label and features) and output 3 columns as result.
 *
 * @param probClassifier the probabilistic classifier to wrap
 * @param uid            stage uid
 * @tparam E spark estimator to wrap
 * @tparam M spark model type returned by spark estimator wrapped
 */
class OpProbabilisticClassifierWrapper[E <: ProbabilisticClassifier[Vector, E, M],
M <: ProbabilisticClassificationModel[Vector, M]]
(
  val probClassifier: E,
  uid: String = UID[OpProbabilisticClassifierWrapper[E, M]]
) extends SwThreeStageBinaryEstimator[RealNN, OPVector, RealNN, OPVector, OPVector, M, E](
  inputParam1Name = SparkMLSharedParamConstants.LabelColName,
  inputParam2Name = SparkMLSharedParamConstants.FeaturesColName,
  outputParam1Name = SparkMLSharedParamConstants.PredictionColName,
  outputParam2Name = SparkMLSharedParamConstants.RawPredictionColName,
  outputParam3Name = SparkMLSharedParamConstants.ProbabilityColName,
  stage1OperationName = probClassifier.getClass.getSimpleName + "_" + SparkMLSharedParamConstants.PredictionColName ,
  stage2OperationName = probClassifier.getClass.getSimpleName + "_" + SparkMLSharedParamConstants.RawPredictionColName,
  stage3OperationName = probClassifier.getClass.getSimpleName + "_" + SparkMLSharedParamConstants.ProbabilityColName,
  // cloning below to prevent parameter changes to the underlying classifier outside the wrapper
  sparkMlStageIn = Option(probClassifier).map(_.copy(ParamMap.empty)),
  uid = uid
) {
  final protected def getSparkStage: E = getSparkMlStage().get
}
