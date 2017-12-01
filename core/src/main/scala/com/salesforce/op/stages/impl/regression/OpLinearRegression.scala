/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.regression

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.OpPredictorWrapper
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}

/**
 * Wrapper around spark ml linear regression for use with OP pipelines
 */
class OpLinearRegression(uid: String = UID[OpLinearRegression])
  extends OpPredictorWrapper[RealNN, RealNN, LinearRegression, LinearRegressionModel](
    predictor = new LinearRegression(),
    uid = uid
){

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
  def setRegParam(value: Double): this.type = {
    getSparkStage.setRegParam(value)
    this
  }

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
   * For 0 < alpha < 1, the penalty is a combination of L1 and L2.
   * Default is 0.0 which is an L2 penalty.
   *
   * @group setParam
   */
  def setElasticNetParam(value: Double): this.type = {
    getSparkStage.setElasticNetParam(value)
    this
  }

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = {
    getSparkStage.setMaxIter(value)
    this
  }

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  def setTol(value: Double): this.type = {
    getSparkStage.setTol(value)
    this
  }

  /**
   * Whether to fit an intercept term.
   * Default is true.
   *
   * @group setParam
   */
  def setFitIntercept(value: Boolean): this.type = {
    getSparkStage.setFitIntercept(value)
    this
  }

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
  def setStandardization(value: Boolean): this.type = {
    getSparkStage.setStandardization(value)
    this
  }

  /**
   * Whether to over-/under-sample training instances according to the given weights in weightCol.
   * If not set or empty String, all instances are treated equally (weight 1.0).
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  def setWeightCol(value: String): this.type = {
    getSparkStage.setWeightCol(value)
    this
  }

  /**
   * Set the solver algorithm used for optimization. In case of linear regression, this can be "l-bfgs", "normal" and
   * "auto".
   * "l-bfgs": Limited-memory BFGS which is a limited-memory quasi-Newton optimization method.
   * "normal": Normal Equation as an analytical solution to the linear regression problem.
   * "auto" (default): solver algorithm is selected automatically. The Normal Equations solver will be used when
   * possible, but this will automatically fall back to iterative optimization methods when needed.
   *
   * @group setParam
   */
  def setSolver(value: String): this.type = {
    getSparkStage.setSolver(value)
    this
  }

  /**
   * Get the regularization parameter.
   *
   */
  def getRegParam: Double = {
    getSparkStage.getRegParam
  }

  /**
   * Get the ElasticNet mixing parameter.
   *
   */
  def getElasticNetParam: Double = {
    getSparkStage.getElasticNetParam
  }

  /**
   * Get the maximum number of iterations.
   *
   */
  def getMaxIter: Int = {
    getSparkStage.getMaxIter
  }

  /**
   * Get the convergence tolerance of iterations.
   *
   */
  def getTol: Double = {
    getSparkStage.getTol
  }

  /**
   * Get the fit intercept boolean parameter
   *
   */
  def getFitIntercept: Boolean = {
    getSparkStage.getFitIntercept
  }

  /**
   * Get the standardization boolean parameter
   *
   */
  def getStandardization: Boolean = {
    getSparkStage.getStandardization
  }

  /**
   * Get the weights in weightCol defining whether to over-/under-sample training instances
   *
   */
  def getWeightCol: String = {
    getSparkStage.getWeightCol
  }

  /**
   * Get the solver algorithm used for optimization
   *
   */
  def getSolver: String = {
    getSparkStage.getSolver
  }
}
