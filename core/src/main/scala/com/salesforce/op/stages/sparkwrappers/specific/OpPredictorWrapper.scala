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
