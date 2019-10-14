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

import com.salesforce.op._
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictionModel, OpPredictorWrapper}
import com.salesforce.op.utils.reflection.ReflectionUtils.reflectMethod
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, OpLinearRegressionParams}

import scala.reflect.runtime.universe.TypeTag


/**
 * Wrapper around spark ml linear regression [[org.apache.spark.ml.regression.LinearRegression]]
 */
class OpLinearRegression(uid: String = UID[OpLinearRegression])
  extends OpPredictorWrapper[LinearRegression, LinearRegressionModel](
    predictor = new LinearRegression(),
    uid = uid
) with OpLinearRegressionParams {

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
   * Set if we should fit the intercept.
   * Default is true.
   *
   * @group setParam
   */
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  /**
   * Whether to standardize the training features before fitting the model.
   * The coefficients of models will be always returned on the original scale,
   * so it will be transparent for users.
   * Default is true.
   *
   * @note With/without standardization, the models should be always converged
   * to the same solution when no regularization is applied. In R's GLMNET package,
   * the default behavior is true as well.
   *
   * @group setParam
   */
  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty.
   * For alpha = 1, it is an L1 penalty.
   * For alpha in (0,1), the penalty is a combination of L1 and L2.
   * Default is 0.0 which is an L2 penalty.
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
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  /**
   * Whether to over-/under-sample training instances according to the given weights in weightCol.
   * If not set or empty, all instances are treated equally (weight 1.0).
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /**
   * Set the solver algorithm used for optimization.
   * In case of linear regression, this can be "l-bfgs", "normal" and "auto".
   *  - "l-bfgs" denotes Limited-memory BFGS which is a limited-memory quasi-Newton
   *    optimization method.
   *  - "normal" denotes using Normal Equation as an analytical solution to the linear regression
   *    problem.  This solver is limited to `LinearRegression.MAX_FEATURES_FOR_NORMAL_SOLVER`.
   *  - "auto" (default) means that the solver algorithm is selected automatically.
   *    The Normal Equations solver will be used when possible, but this will automatically fall
   *    back to iterative optimization methods when needed.
   *
   * @group setParam
   */
  def setSolver(value: String): this.type = {
    require(Set("auto", "l-bfgs", "normal").contains(value),
      s"Solver $value was not supported. Supported options: auto, l-bfgs, normal")
    set(solver, value)
  }
  setDefault(solver -> "auto")

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
   * Set the type of loss function to be optimized.
   * Supported options: squaredError (https://en.wikipedia.org/wiki/Mean_squared_error)
   * and "huber" (https://en.wikipedia.org/wiki/Huber_loss).
   * Default is squaredError.
   *
   * @group setParam
   */
  def setLossType(value: String): this.type = set(loss, value)
  setDefault(loss -> "squaredError")

  /**
   * The shape parameter to control the amount of robustness. Must be &gt; 1.0.
   * At larger values of epsilon, the huber criterion becomes more similar to least squares
   * regression; for small values of epsilon, the criterion is more similar to L1 regression.
   * Default is 1.35 to get as much robustness as possible while retaining
   * 95% statistical efficiency for normally distributed data. It matches sklearn
   * HuberRegressor and is "M" from <a href="http://statweb.stanford.edu/~owen/reports/hhu.pdf">
   * A robust hybrid of lasso and ridge regression</a>.
   * Only valid when "loss" is "huber".
   *
   * @group expertParam
   */
  def setEpsilon(value: Double): this.type = set(epsilon, value)
  setDefault(epsilon -> 1.35)

}


/**
 * Class that takes in a spark LinearRegressionModel and wraps it into an OP model which returns a
 * Prediction feature
 * @param sparkModel model to wrap
 * @param uid uid to give stage
 * @param operationName unique name of the operation this stage performs
 */
class OpLinearRegressionModel
(
  sparkModel: LinearRegressionModel,
  uid: String = UID[OpLinearRegressionModel],
  operationName: String = classOf[LinearRegression].getSimpleName
)(
  implicit tti1: TypeTag[RealNN],
  tti2: TypeTag[OPVector],
  tto: TypeTag[Prediction],
  ttov: TypeTag[Prediction#Value]
) extends OpPredictionModel[LinearRegressionModel](
  sparkModel = sparkModel, uid = uid, operationName = operationName
) {
  @transient lazy val predictMirror = reflectMethod(getSparkMlStage().get, "predict")
}

