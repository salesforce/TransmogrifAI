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

