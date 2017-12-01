/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.UID
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.OpProbabilisticClassifierWrapper
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}

/**
 * Wrapper around spark ml logistic regression for use with OP pipelines
 */
class OpLogisticRegression(uid: String = UID[OpLogisticRegression])
  extends OpProbabilisticClassifierWrapper[LogisticRegression, LogisticRegressionModel](
    new LogisticRegression(),
    uid = uid
  ) {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  /**
   * Set thresholds in multiclass (or binary) classification to adjust the probability of
   * predicting each class. Array must have length equal to the number of classes, with values >= 0.
   * The class with largest value p/t is predicted, where p is the original probability of that
   * class and t is the class' threshold.
   *
   * @group setParam
   */
  def setThresholds(value: Array[Double]): this.type = {
    getSparkStage.setThresholds(value)
    this
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

}
