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
