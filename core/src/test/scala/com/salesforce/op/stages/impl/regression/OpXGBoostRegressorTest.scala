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

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.PredictionEquality
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpPredictorWrapperModel}
import com.salesforce.op.test._
import ml.dmlc.xgboost4j.scala.spark.{OpXGBoostQuietLogging, XGBoostRegressionModel, XGBoostRegressor}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpXGBoostRegressorTest extends OpEstimatorSpec[Prediction, OpPredictorWrapperModel[XGBoostRegressionModel],
  OpPredictorWrapper[XGBoostRegressor, XGBoostRegressionModel]]
  with PredictionEquality with OpXGBoostQuietLogging {

  override def specName: String = classOf[OpXGBoostRegressor].getSimpleName

  val rawData = Seq(
    (10.0, Vectors.dense(1.0, 4.3, 1.3)),
    (20.0, Vectors.dense(2.0, 0.3, 0.1)),
    (30.0, Vectors.dense(3.0, 3.9, 4.3)),
    (40.0, Vectors.dense(4.0, 1.3, 0.9)),
    (50.0, Vectors.dense(5.0, 4.7, 1.3))
  ).map { case (l, v) => l.toRealNN -> v.toOPVector }

  val (inputData, label, features) = TestFeatureBuilder("label", "features", rawData)

  val estimator = new OpXGBoostRegressor().setInput(label.copy(isResponse = true), features)
  estimator.setSilent(1)

  val expectedResult = Seq(
    Prediction(1.9250000715255737),
    Prediction(8.780000686645508),
    Prediction(8.780000686645508),
    Prediction(8.780000686645508),
    Prediction(8.780000686645508)
  )

  it should "allow the user to set the desired spark parameters" in {
    estimator.setMaxDepth(18).setBaseScore(0.12345).setSkipDrop(0.6234)
    estimator.fit(inputData)
    estimator.predictor.getMaxDepth shouldBe 18
    estimator.predictor.getBaseScore shouldBe 0.12345
    estimator.predictor.getSkipDrop shouldBe 0.6234

  }
}
