/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.sparkwrappers.specific.OpBinaryEstimatorWrapper
import org.apache.spark.ml.regression.{IsotonicRegression, IsotonicRegressionModel}

/**
 * Isotonic regression calibrator.
 *
 * Uses [[org.apache.spark.ml.regression.IsotonicRegression]], which supports only univariate
 * (single feature) models.
 */
class IsotonicRegressionCalibrator(uid: String = UID[IsotonicRegressionCalibrator])
  extends OpBinaryEstimatorWrapper[RealNN, RealNN, RealNN,
    IsotonicRegression, IsotonicRegressionModel](
    estimator = new IsotonicRegression().setIsotonic(true),
    uid = uid
  ) {

  /**
   * Param for whether the output sequence should be isotonic/increasing (true) or
   * antitonic/decreasing (false).
   * Default: true
   *
   * @param value
   * @return the current IsotonicRegressionCalibrator instance
   */
  def setIsotonic(value: Boolean): this.type = {
    getSparkStage.setIsotonic(value)
    this
  }
}
