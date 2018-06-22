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

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.PredictionEquality
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpPredictorWrapperModel}
import com.salesforce.op.test._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpRandomForestRegressorTest extends OpEstimatorSpec[Prediction,
  OpPredictorWrapperModel[RandomForestRegressionModel],
  OpPredictorWrapper[RandomForestRegressor, RandomForestRegressionModel]] with PredictionEquality {

  val (inputData, rawLabel, features) = TestFeatureBuilder(
    Seq[(RealNN, OPVector)](
      (10.0.toRealNN, Vectors.dense(1.0, 4.3, 1.3).toOPVector),
      (20.0.toRealNN, Vectors.dense(2.0, 0.3, 0.1).toOPVector),
      (30.0.toRealNN, Vectors.dense(3.0, 3.9, 4.3).toOPVector),
      (40.0.toRealNN, Vectors.dense(4.0, 1.3, 0.9).toOPVector),
      (50.0.toRealNN, Vectors.dense(5.0, 4.7, 1.3).toOPVector)
    )
  )
  val label = rawLabel.copy(isResponse = true)
  val estimator = new OpRandomForestRegressor().setInput(label, features)

  val expectedResult = Seq(
    Prediction(20.0),
    Prediction(23.5),
    Prediction(31.5),
    Prediction(35.5),
    Prediction(37.0)
  )

  it should "allow the user to set the desired spark parameters" in {
    estimator
      .setMaxDepth(7)
      .setMaxBins(3)
      .setMinInstancesPerNode(2)
      .setMinInfoGain(0.1)
      .setSeed(42L)
    estimator.fit(inputData)

    estimator.predictor.getMaxDepth shouldBe 7
    estimator.predictor.getMaxBins shouldBe 3
    estimator.predictor.getMinInstancesPerNode shouldBe 2
    estimator.predictor.getMinInfoGain shouldBe 0.1
    estimator.predictor.getSeed shouldBe 42L

  }
}
