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
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpProbabilisticClassifierModel}
import com.salesforce.op.utils.reflection.ReflectionUtils.reflectMethod
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, OpLogisticRegressionParams}
import org.apache.spark.ml.linalg.{Matrix, Vector}

import scala.reflect.runtime.universe.TypeTag

/**
 * Wrapper around spark ml logistic regression [[org.apache.spark.ml.classification.LogisticRegression]]
 */
class OpLogisticRegression(uid: String = UID[OpLogisticRegression])
  extends OpPredictorWrapper[LogisticRegression, LogisticRegressionModel](
    predictor = new LogisticRegression(),
    uid = uid
  ) with OpLogisticRegressionParams {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  /**
   * Set the regularization parameter.
   * Default is 0.0.
   *
   * @group setParam
   */
  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty.
   * For alpha = 1, it is an L1 penalty.
   * For alpha in (0,1), the penalty is a combination of L1 and L2.
   * Default is 0.0 which is an L2 penalty.
   *
   * Note: Fitting under bound constrained optimization only supports L2 regularization,
   * so throws exception if this param is non-zero value.
   *
   * @group setParam
   */
  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
  setDefault(elasticNetParam -> 0.0)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy at the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  /**
   * Whether to fit an intercept term.
   * Default is true.
   *
   * @group setParam
   */
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  /**
   * Sets the value of param [[family]].
   * Default is "auto".
   *
   * @group setParam
   */
  def setFamily(value: String): this.type = set(family, value)
  setDefault(family -> "auto")

  /**
   * Whether to standardize the training features before fitting the model.
   * The coefficients of models will be always returned on the original scale,
   * so it will be transparent for users. Note that with/without standardization,
   * the models should be always converged to the same solution when no regularization
   * is applied. In R's GLMNET package, the default behavior is true as well.
   * Default is true.
   *
   * @group setParam
   */
  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  override def setThreshold(value: Double): this.type = super.setThreshold(value)


  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  override def setThresholds(value: Array[Double]): this.type = super.setThresholds(value)

  /**
   * Suggested depth for treeAggregate (greater than or equal to 2).
   * If the dimensions of features or the number of partitions are large,
   * this param could be adjusted to a larger size.
   * Default is 2.
   *
   * @group expertSetParam
   */
  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
  setDefault(aggregationDepth -> 2)

  /**
   * Set the lower bounds on coefficients if fitting under bound constrained optimization.
   *
   * @group expertSetParam
   */
  def setLowerBoundsOnCoefficients(value: Matrix): this.type = set(lowerBoundsOnCoefficients, value)

  /**
   * Set the upper bounds on coefficients if fitting under bound constrained optimization.
   *
   * @group expertSetParam
   */
  def setUpperBoundsOnCoefficients(value: Matrix): this.type = set(upperBoundsOnCoefficients, value)

  /**
   * Set the lower bounds on intercepts if fitting under bound constrained optimization.
   *
   * @group expertSetParam
   */
  def setLowerBoundsOnIntercepts(value: Vector): this.type = set(lowerBoundsOnIntercepts, value)

  /**
   * Set the upper bounds on intercepts if fitting under bound constrained optimization.
   *
   * @group expertSetParam
   */
  def setUpperBoundsOnIntercepts(value: Vector): this.type = set(upperBoundsOnIntercepts, value)

}


/**
 * Class that takes in a spark LogisticRegressionModel and wraps it into an OP model which returns a
 * Prediction feature
 *
 * @param sparkModel    model to wrap
 * @param uid           uid to give stage
 * @param operationName unique name of the operation this stage performs
 */
class OpLogisticRegressionModel
(
  sparkModel: LogisticRegressionModel,
  uid: String = UID[OpLogisticRegressionModel],
  operationName: String = classOf[LogisticRegression].getSimpleName
)(
  implicit tti1: TypeTag[RealNN],
  tti2: TypeTag[OPVector],
  tto: TypeTag[Prediction],
  ttov: TypeTag[Prediction#Value]
) extends OpProbabilisticClassifierModel[LogisticRegressionModel](
  sparkModel = sparkModel, uid = uid, operationName = operationName
) {
  @transient lazy val predictRawMirror = getSparkOrLocalMethod("predictRaw", "predictRaw")
  @transient lazy val raw2probabilityMirror = getSparkOrLocalMethod("raw2probability", "rawToProbability")
  @transient lazy val probability2predictionMirror = getSparkOrLocalMethod("probability2prediction",
    "probabilityToPrediction")
}
