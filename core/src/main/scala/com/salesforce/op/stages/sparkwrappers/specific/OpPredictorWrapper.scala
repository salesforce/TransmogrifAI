/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

// scalastyle:off
package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.UID
import com.salesforce.op.features.types.{FeatureType, OPVector}
import com.salesforce.op.stages.sparkwrappers.generic.SwBinaryEstimator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{PredictionModel, Predictor, SparkMLSharedParamConstants}

import scala.reflect.runtime.universe.TypeTag

/**
 * Wraps a spark ML predictor.  Predictors represent supervised learning algorithms (regression and classification) in
 * spark ML that inherit from [[Predictor]], examples of which include:
 * [[org.apache.spark.ml.regression.RandomForestRegressor]],
 * [[org.apache.spark.ml.regression.GBTRegressor]], [[org.apache.spark.ml.classification.GBTClassifier]],
 * [[org.apache.spark.ml.regression.DecisionTreeRegressor]],
 * [[org.apache.spark.ml.classification.MultilayerPerceptronClassifier]],
 * [[org.apache.spark.ml.regression.LinearRegression]],
 * and [[org.apache.spark.ml.regression.GeneralizedLinearRegression]].
 * Their defining characteristic is that they output a model which takes in 2 columns as input (labels and features)
 * and output one column as result.
 * NOTE: Probabilistic classifiers contain additional output information, and so there is a specific wrapper
 * for that kind of classifier see: [[OpProbabilisticClassifierWrapper]]
 *
 * @param predictor the predictor to wrap
 * @param uid       stage uid
 * @tparam I the type of the transformation input feature
 * @tparam O the type of the transformation output feature
 * @tparam E spark estimator to wrap
 * @tparam M spark model type returned by spark estimator wrapped
 */
class OpPredictorWrapper[I <: FeatureType, O <: FeatureType, E <: Predictor[Vector, E, M],
M <: PredictionModel[Vector, M]]
(
  val predictor: E,
  uid: String = UID[OpPredictorWrapper[I, O, E, M]]
)(
  implicit tti1: TypeTag[I],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends SwBinaryEstimator[I, OPVector, O, M, E](
  inputParam1Name = SparkMLSharedParamConstants.LabelColName,
  inputParam2Name = SparkMLSharedParamConstants.FeaturesColName,
  outputParamName = SparkMLSharedParamConstants.PredictionColName,
  operationName = predictor.getClass.getSimpleName,
  // cloning below to prevent parameter changes to the underlying classifier outside the wrapper
  sparkMlStageIn = Option(predictor).map(_.copy(ParamMap.empty)),
  uid = uid
) {
  final protected def getSparkStage: E = getSparkMlStage().get
}
