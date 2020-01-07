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

import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector

import scala.reflect.runtime.universe._

/**
 * Class that takes in a spark PredictionModel and wraps it into an OP model which returns a
 * Prediction feature
 *
 * @param sparkModel    model to wrap
 * @param uid           uid to give stage
 * @param operationName unique name of the operation this stage performs
 * @tparam T type of the model to wrap
 */
abstract class OpPredictionModel[T <: PredictionModel[Vector, T]]
(
  sparkModel: T,
  uid: String,
  operationName: String
) extends OpPredictorWrapperModel[T](uid = uid, operationName = operationName, sparkModel = sparkModel) {

  /**
   * Predict label for the given features
   */
  @transient protected lazy val predict: Vector => Double = getSparkMlStage().getOrElse(
    throw new RuntimeException("Could not find the wrapped Spark stage.")
  ).predict(_)

  /**
   * Function used to convert input to output
   */
  override def transformFn: (RealNN, OPVector) => Prediction = (label, features) =>
    Prediction(prediction = predict(features.value))

}
