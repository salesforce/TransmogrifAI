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

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpProbabilisticClassifierModel}
import com.salesforce.op.utils.reflection.ReflectionUtils.reflectMethod
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel, OpNaiveBayesParams}

import scala.reflect.runtime.universe.TypeTag

/**
 * Wrapper for spark Naive Bayes [[org.apache.spark.ml.classification.NaiveBayesModel]]
 * @param uid       stage uid
 */
class OpNaiveBayes(uid: String = UID[OpNaiveBayes])
  extends OpPredictorWrapper[NaiveBayes, NaiveBayesModel](
    predictor = new NaiveBayes(),
    uid = uid
  ) with OpNaiveBayesParams {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  /**
   * Set the smoothing parameter.
   * Default is 1.0.
   * @group setParam
   */
  def setSmoothing(value: Double): this.type = set(smoothing, value)
  setDefault(smoothing -> 1.0)

  /**
   * Set the model type using a string (case-sensitive).
   * Supported options: "multinomial" and "bernoulli".
   * Default is "multinomial"
   * @group setParam
   */
  def setModelType(value: String): this.type = set(modelType, value)
  setDefault(modelType -> "multinomial")

  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  def setWeightCol(value: String): this.type = set(weightCol, value)
}


/**
 * Class that takes in a spark NaiveBayesModel and wraps it into an OP model which returns a
 * Prediction feature
 *
 * @param sparkModel    model to wrap
 * @param uid           uid to give stage
 * @param operationName unique name of the operation this stage performs
 */
class OpNaiveBayesModel
(
  sparkModel: NaiveBayesModel,
  uid: String = UID[OpNaiveBayesModel],
  operationName: String = classOf[NaiveBayes].getSimpleName
)(
  implicit tti1: TypeTag[RealNN],
  tti2: TypeTag[OPVector],
  tto: TypeTag[Prediction],
  ttov: TypeTag[Prediction#Value]
) extends OpProbabilisticClassifierModel[NaiveBayesModel](
  sparkModel = sparkModel, uid = uid, operationName = operationName
) {
  @transient lazy val predictRawMirror = reflectMethod(getSparkMlStage().get, "predictRaw")
  @transient lazy val raw2probabilityMirror = reflectMethod(getSparkMlStage().get, "raw2probability")
  @transient lazy val probability2predictionMirror =
    reflectMethod(getSparkMlStage().get, "probability2prediction")
}


