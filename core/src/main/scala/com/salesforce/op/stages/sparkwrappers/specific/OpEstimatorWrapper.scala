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

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.sparkwrappers.generic.SwUnaryEstimator
import org.apache.spark.ml._
import org.apache.spark.ml.param.ParamMap

import scala.reflect.runtime.universe.TypeTag

/**
 * Wraps a spark ML estimator.  This wrapper is meant for Estimators not already covered by more specific
 * wrappers such as: [[OpPredictorWrapper]].
 * Examples of estimators meant to be wrapped with OpEstimatorWrapper include MinMaxScaler, IDF, VectorIndexer,
 * CountVectorizer, QuantileDiscretizer, StandardScaler, PCA, MaxAbsScaler, Word2Vec, etc.
 * Their defining characteristic is that they output a Model which takes in one column as input and output
 * one column as a result.
 *
 * @param estimator the estimator to wrap
 * @param uid       stage uid
 * @tparam I the type of the transformation input feature
 * @tparam O the type of the transformation output feature
 * @tparam E spark estimator to wrap
 * @tparam M spark model type returned by spark estimator wrapped
 */
class OpEstimatorWrapper[I <: FeatureType, O <: FeatureType, E <: Estimator[M], M <: Model[M]]
(
  val estimator: E,
  uid: String = UID[OpEstimatorWrapper[I, O, E, M]]
)(
  implicit tti: TypeTag[I],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends SwUnaryEstimator[I, O, M, E](
  inputParamName = SparkMLSharedParamConstants.InputColName,
  outputParamName = SparkMLSharedParamConstants.OutputColName,
  operationName = estimator.getClass.getSimpleName,
  // cloning below to prevent parameter changes to the underlying classifier outside the wrapper
  sparkMlStageIn = Option(estimator).map(_.copy(ParamMap.empty).asInstanceOf[E]),
  uid = uid
)

