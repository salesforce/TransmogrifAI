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

import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.base.binary.OpTransformer2
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.regression._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression._
import org.apache.spark.ml.{Model, PredictionModel}

/**
 * Allows conversion from spark models to models that follow the OP convention of having a
 * transformFn that can be called on a single row rather than the whole dataframe
 */
object SparkModelConverter {

  /**
   * Converts supported spark model of type PredictionModel[Vector, T] to an OP model
   * @param model model to convert
   * @param uid uid to give converted model
   * @tparam T type of model to convert
   * @return Op Binary Model which will produce the same values put into a Prediction return feature
   */
  def toOP[T <: PredictionModel[Vector, T]](
    model: T,
    uid: String
  ): OpPredictorWrapperModel[T] = {
    toOPUnchecked(model, uid).asInstanceOf[OpPredictorWrapperModel[T]]
  }

  /**
   * Converts supported spark model of type PredictionModel[Vector, T] to an OP model
   * @param model model to convert
   * @tparam T type of model to convert
   * @return Op Binary Model which will produce the same values put into a Prediction return feature
   */
  // TODO remove when loco and model selector are updated
  def toOPUnchecked[T <: Model[_]](model: T): OpTransformer2[RealNN, OPVector, Prediction] =
    toOPUnchecked(model, model.uid)

  /**
   * Converts supported spark model of type PredictionModel[Vector, T] to an OP model
   * @param model model to convert
   * @param uid uid to give converted model
   * @tparam T type of model to convert
   * @return Op Binary Model which will produce the same values put into a Prediction return feature
   */
  // TODO remove when loco and model selector are updated
  def toOPUnchecked[T <: Model[_]](
    model: T,
    uid: String
  ): OpTransformer2[RealNN, OPVector, Prediction] = {
    model match {
      case m: LogisticRegressionModel => new OpLogisticRegressionModel(m, uid = uid)
      case m: RandomForestClassificationModel => new OpRandomForestClassificationModel(m, uid = uid)
      case m: NaiveBayesModel => new OpNaiveBayesModel(m, uid)
      case m: DecisionTreeClassificationModel => new OpDecisionTreeClassificationModel(m, uid = uid)
      case m: GBTClassificationModel => new OpGBTClassificationModel(m, uid = uid)
      case m: LinearSVCModel => new OpLinearSVCModel(m, uid = uid)
      case m: MultilayerPerceptronClassificationModel => new OpMultilayerPerceptronClassificationModel(m, uid = uid)
      case m: LinearRegressionModel => new OpLinearRegressionModel(m, uid = uid)
      case m: RandomForestRegressionModel => new OpRandomForestRegressionModel(m, uid = uid)
      case m: GBTRegressionModel => new OpGBTRegressionModel(m, uid = uid)
      case m: DecisionTreeRegressionModel => new OpDecisionTreeRegressionModel(m, uid = uid)
      case m: GeneralizedLinearRegressionModel => new OpGeneralizedLinearRegressionModel(m, uid = uid)
      case m => throw new RuntimeException(s"model conversion not implemented for model $m")
    }
  }

}
