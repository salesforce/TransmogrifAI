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

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpPredictorWrapperModel}
import ml.combust.mleap.core.regression.{GeneralizedLinearRegressionModel => MleapGeneralizedLinearRegressionModel}
import com.salesforce.op.utils.reflection.ReflectionUtils.reflectMethod
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, GeneralizedLinearRegressionModel, OpGeneralizedLinearRegressionParams}
import org.apache.spark.ml.linalg.Vector

import scala.reflect.runtime.universe.TypeTag

/**
 * Wrapper for spark Generalized Regression [[org.apache.spark.ml.regression.GeneralizedLinearRegression]]
 * @param uid       stage uid
 */
class OpGeneralizedLinearRegression(uid: String = UID[OpGeneralizedLinearRegression])
  extends OpPredictorWrapper[GeneralizedLinearRegression, GeneralizedLinearRegressionModel](
    predictor = new GeneralizedLinearRegression(),
    uid = uid
  ) with OpGeneralizedLinearRegressionParams {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  /**
   * Sets the value of param [[family]].
   * Default is "gaussian".
   *
   * @group setParam
   */
  def setFamily(value: String): this.type = set(family, value)
  setDefault(family -> "gaussian")

  /**
   * Sets the value of param [[variancePower]].
   * Used only when family is "tweedie".
   * Default is 0.0, which corresponds to the "gaussian" family.
   *
   * @group setParam
   */
  def setVariancePower(value: Double): this.type = set(variancePower, value)
  setDefault(variancePower -> 0.0)

  /**
   * Sets the value of param [[linkPower]].
   * Used only when family is "tweedie".
   *
   * @group setParam
   */
  def setLinkPower(value: Double): this.type = set(linkPower, value)

  /**
   * Sets the value of param [[link]].
   * Used only when family is not "tweedie".
   *
   * @group setParam
   */
  def setLink(value: String): this.type = set(link, value)

  /**
   * Sets if we should fit the intercept.
   * Default is true.
   *
   * @group setParam
   */
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)

  /**
   * Sets the maximum number of iterations (applicable for solver "irls").
   * Default is 25.
   *
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 25)

  /**
   * Sets the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  /**
   * Sets the regularization parameter for L2 regularization.
   * The regularization term is
   * <blockquote>
   *    $$
   *    0.5 * regParam * L2norm(coefficients)^2
   *    $$
   * </blockquote>
   * Default is 0.0.
   *
   * @group setParam
   */
  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * Default is not set, so all instances have weight one.
   * In the Binomial family, weights correspond to number of trials and should be integer.
   * Non-integer weights are rounded to integer in AIC calculation.
   *
   * @group setParam
   */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /**
   * Sets the solver algorithm used for optimization.
   * Currently only supports "irls" which is also the default solver.
   *
   * @group setParam
   */
  def setSolver(value: String): this.type = set(solver, value)
  setDefault(solver -> "irls")

  /**
   * Sets the link prediction (linear predictor) column name.
   *
   * @group setParam
   */
  def setLinkPredictionCol(value: String): this.type = set(linkPredictionCol, value)

}



/**
 * Class that takes in a spark GeneralizedLinearRegressionModel and wraps it into an OP model which returns a
 * Prediction feature
 *
 * @param sparkModel    model to wrap
 * @param uid           uid to give stage
 * @param operationName unique name of the operation this stage performs
 */
class OpGeneralizedLinearRegressionModel
(
  sparkModel: GeneralizedLinearRegressionModel,
  uid: String = UID[GeneralizedLinearRegressionModel],
  operationName: String = classOf[GeneralizedLinearRegression].getSimpleName
)(
  implicit tti1: TypeTag[RealNN],
  tti2: TypeTag[OPVector],
  tto: TypeTag[Prediction],
  ttov: TypeTag[Prediction#Value]
) extends OpPredictorWrapperModel[GeneralizedLinearRegressionModel](uid = uid, operationName = operationName,
  sparkModel = sparkModel) {

  @transient lazy private val predictLinkSpark = getSparkMlStage()
    .map(s => reflectMethod(s, "predictLink", argsCount = Option(2)))
  @transient lazy private val predictLinkLocal = getLocalMlStage()
    .map(s => s.model.asInstanceOf[MleapGeneralizedLinearRegressionModel].predictLink(_))

  /**
   * Function used to convert input to output
   */
  override def transformFn: (RealNN, OPVector) => Prediction = (_, features) => {
    val raw: Double = predictLinkSpark.map( p => p(features.value, 0.0).asInstanceOf[Double] )
      .orElse( predictLinkLocal.map(p => p(features.value)) )
      .getOrElse(throw new RuntimeException("Failed to find link function in local or spark models"))
    val pred: Double = predict(features.value)
    Prediction(prediction = pred, rawPrediction = raw)
  }
}
