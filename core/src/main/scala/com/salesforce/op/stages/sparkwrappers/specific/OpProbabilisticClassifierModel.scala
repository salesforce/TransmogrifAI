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

import com.salesforce.op.features.types._
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.Vector

import scala.reflect.runtime.universe._

/**
 * Class that takes in a spark ProbabilisticClassifierModel and wraps it into an OP model which returns a
 * Prediction feature
 *
 * @param sparkModel    model to wrap
 * @param uid           uid to give stage
 * @param operationName unique name of the operation this stage performs
 * @tparam T type of the model to wrap
 */
abstract class OpProbabilisticClassifierModel[T <: ProbabilisticClassificationModel[Vector, T]]
(
  sparkModel: T,
  uid: String,
  operationName: String
) extends OpPredictorWrapperModel[T](uid = uid, operationName = operationName, sparkModel = sparkModel) {

  protected def predictRawMirror: MethodMirror
  protected def raw2probabilityMirror: MethodMirror
  protected def probability2predictionMirror: MethodMirror

  protected def predictRaw(features: Vector): Vector = predictRawMirror.apply(features).asInstanceOf[Vector]
  protected def raw2probability(raw: Vector): Vector = raw2probabilityMirror.apply(raw).asInstanceOf[Vector]
  protected def probability2prediction(prob: Vector): Double =
    probability2predictionMirror.apply(prob).asInstanceOf[Double]

  /**
   * Function used to convert input to output
   */
  override def transformFn: (RealNN, OPVector) => Prediction = (label, features) => {
    val raw = predictRaw(features.value)
    val prob = raw2probability(raw)
    val pred = probability2prediction(prob)

    Prediction(rawPrediction = raw, probability = prob, prediction = pred)
  }

}
