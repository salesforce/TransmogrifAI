/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.sparkwrappers.generic.SwBinaryEstimator
import org.apache.spark.ml._
import org.apache.spark.ml.param.ParamMap

import scala.reflect.runtime.universe.TypeTag

/**
 * Wraps a spark ML estimator.  This class is meant for Predictor-like Spark algorithms, but which don't
 * inherit from [[Predictor]], for whatever reason.  Examples of which include:
 * IsotonicRegression, AFTSurvivalRegression, OneVsRest.
 * Their defining characteristic is that they output a model which takes in 2 columns as input (labels and features)
 * and output one column as result, but don't inherit from [[Predictor]] (if it did,
 * should use [[OpPredictorWrapper]] instead).
 *
 * @param estimator spark estimator to wrap
 * @param uid       stage uid
 * @param tti1      type tag for first input
 * @param tti2      type tag for second input
 * @param tto       type tag for input
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam O  output feature type
 * @tparam E  spark estimator to wrap
 * @tparam M  spark model type returned by spark estimator wrapped
 */
class OpBinaryEstimatorWrapper[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType, E <: Estimator[M], M <: Model[M]]
(
  val estimator: E,
  uid: String = UID[OpBinaryEstimatorWrapper[I1, I2, O, E, M]]
)(implicit tti1: TypeTag[I1],
  tti2: TypeTag[I2],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends SwBinaryEstimator[I1, I2, O, M, E](
  inputParam1Name = SparkMLSharedParamConstants.LabelColName,
  inputParam2Name = SparkMLSharedParamConstants.FeaturesColName,
  outputParamName = SparkMLSharedParamConstants.PredictionColName,
  operationName = estimator.getClass.getSimpleName,
  // cloning below to prevent parameter changes to the underlying classifier outside the wrapper
  sparkMlStageIn = Option(estimator).map(_.copy(ParamMap.empty).asInstanceOf[E]),
  uid = uid
) {
  final protected def getSparkStage: E = getSparkMlStage().get
}

