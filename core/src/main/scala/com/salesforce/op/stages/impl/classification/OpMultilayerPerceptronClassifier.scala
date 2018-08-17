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

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictionModel, OpPredictorWrapper}
import com.salesforce.op.utils.reflection.ReflectionUtils.reflectMethod
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier, OpMultilayerPerceptronClassifierParams}
import org.apache.spark.ml.linalg.Vector

import scala.reflect.runtime.universe.TypeTag

/**
 * Wrapper for spark MultiLayerPerceptronClassifier
 * [[org.apache.spark.ml.classification.MultilayerPerceptronClassifier]]
 * @param uid       stage uid
 */
class OpMultilayerPerceptronClassifier(uid: String = UID[OpMultilayerPerceptronClassifier])
  extends OpPredictorWrapper[MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel](
    predictor = new MultilayerPerceptronClassifier(),
    uid = uid
  ) with OpMultilayerPerceptronClassifierParams {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  /**
   * Sets the value of param [[layers]].
   *
   * @group setParam
   */
  def setLayers(value: Array[Int]): this.type = set(layers, value)

  /**
   * Sets the value of param [[blockSize]].
   * Default is 128.
   *
   * @group expertSetParam
   */
  def setBlockSize(value: Int): this.type = set(blockSize, value)

  /**
   * Sets the value of param [[solver]].
   * Default is "l-bfgs".
   *
   * @group expertSetParam
   */
  def setSolver(value: String): this.type = set(solver, value)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)

  /**
   * Set the seed for weights initialization if weights are not set
   *
   * @group setParam
   */
  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Sets the value of param [[initialWeights]].
   *
   * @group expertSetParam
   */
  def setInitialWeights(value: Vector): this.type = set(initialWeights, value)

  /**
   * Sets the value of param [[stepSize]] (applicable only for solver "gd").
   * Default is 0.03.
   *
   * @group setParam
   */
  def setStepSize(value: Double): this.type = set(stepSize, value)
}


/**
 * Class that takes in a spark MultilayerPerceptronClassificationModel and wraps it into an OP model which returns a
 * Prediction feature
 *
 * @param sparkModel    model to wrap
 * @param uid           uid to give stage
 * @param operationName unique name of the operation this stage performs
 */
// TODO in next release of spark this will be a probabilistic classifier
class OpMultilayerPerceptronClassificationModel
(
  sparkModel: MultilayerPerceptronClassificationModel,
  uid: String = UID[OpMultilayerPerceptronClassificationModel],
  operationName: String = classOf[MultilayerPerceptronClassifier].getSimpleName
)(
  implicit tti1: TypeTag[RealNN],
  tti2: TypeTag[OPVector],
  tto: TypeTag[Prediction],
  ttov: TypeTag[Prediction#Value]
) extends OpPredictionModel[MultilayerPerceptronClassificationModel](
  sparkModel = sparkModel, uid = uid, operationName = operationName
) {
  @transient lazy val predictMirror = reflectMethod(getSparkMlStage().get, "predict")
}

