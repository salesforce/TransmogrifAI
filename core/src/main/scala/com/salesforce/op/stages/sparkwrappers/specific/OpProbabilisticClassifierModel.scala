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

import com.salesforce.op.features.types._
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.Vector

import scala.reflect.ClassTag

/**
 * Class that takes in a spark ProbabilisticClassifierModel and wraps it into an OP model which returns a
 * Prediction feature
 *
 * @param sparkModel    model to wrap
 * @param uid           uid to give stage
 * @param operationName unique name of the operation this stage performs
 * @tparam T type of the model to wrap
 */
abstract class OpProbabilisticClassifierModel[T <: ProbabilisticClassificationModel[Vector, T] : ClassTag]
(
  sparkModel: T,
  uid: String,
  operationName: String
) extends OpPredictorWrapperModel[T](uid = uid, operationName = operationName, sparkModel = sparkModel) {

  @transient private lazy val predictRawMirror = getSparkOrLocalMethod("predictRaw", "predictRaw")
  @transient private lazy val raw2probabilityMirror = getSparkOrLocalMethod("raw2probability", "rawToProbability")
  @transient private lazy val probability2predictionMirror = getSparkOrLocalMethod("probability2prediction",
    "probabilityToPrediction")

  protected def predictRaw(features: Vector): Vector = predictRawMirror.apply(features).asInstanceOf[Vector]
  protected def raw2probability(raw: Vector): Vector = raw2probabilityMirror.apply(raw).asInstanceOf[Vector]
  protected def probability2prediction(prob: Vector): Double =
    probability2predictionMirror.apply(prob).asInstanceOf[Double]

  /**
   * Function used to convert input to output
   */
  override def transformFn: (RealNN, OPVector) => Prediction = (_, features) => {
    val raw = predictRaw(features.value)
    val prob = raw2probability(raw)
    val pred = probability2prediction(prob)

    Prediction(rawPrediction = raw, probability = prob, prediction = pred)
  }

}
