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

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.PredictionEquality
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpPredictorWrapperModel}
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import ml.dmlc.xgboost4j.scala.spark.{OpXGBoostQuietLogging, XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpXGBoostClassifierTest extends OpEstimatorSpec[Prediction, OpPredictorWrapperModel[XGBoostClassificationModel],
  OpPredictorWrapper[XGBoostClassifier, XGBoostClassificationModel]]
  with PredictionEquality with OpXGBoostQuietLogging {

  override def specName: String = Spec[OpXGBoostClassifier]

  val rawData = Seq(
    1.0 -> Vectors.dense(12.0, 4.3, 1.3),
    0.0 -> Vectors.dense(0.0, 0.3, 0.1),
    0.0 -> Vectors.dense(1.0, 3.9, 4.3),
    1.0 -> Vectors.dense(10.0, 1.3, 0.9),
    1.0 -> Vectors.dense(15.0, 4.7, 1.3),
    0.0 -> Vectors.dense(0.5, 0.9, 10.1),
    1.0 -> Vectors.dense(11.5, 2.3, 1.3),
    0.0 -> Vectors.dense(0.1, 3.3, 0.1)
  ).map { case (l, v) => l.toRealNN -> v.toOPVector }

  val (inputData, label, features) = TestFeatureBuilder("label", "features", rawData)

  val estimator = new OpXGBoostClassifier().setInput(label.copy(isResponse = true), features)
  estimator.setSilent(1)

  val expectedResult = Seq(
    Prediction(1.0, Array(0.6200000047683716), Array(0.3799999952316284, 0.6200000047683716)),
    Prediction(0.0, Array(0.3799999952316284), Array(0.6200000047683716, 0.3799999952316284)),
    Prediction(0.0, Array(0.3799999952316284), Array(0.6200000047683716, 0.3799999952316284)),
    Prediction(1.0, Array(0.6200000047683716), Array(0.3799999952316284, 0.6200000047683716)),
    Prediction(1.0, Array(0.6200000047683716), Array(0.3799999952316284, 0.6200000047683716)),
    Prediction(0.0, Array(0.3799999952316284), Array(0.6200000047683716, 0.3799999952316284)),
    Prediction(1.0, Array(0.6200000047683716), Array(0.3799999952316284, 0.6200000047683716)),
    Prediction(0.0, Array(0.3799999952316284), Array(0.6200000047683716, 0.3799999952316284))
  )

  it should "allow the user to set the desired spark parameters" in {
    estimator.setAlpha(0.872).setEta(0.99912)
    estimator.fit(inputData)
    estimator.predictor.getAlpha shouldBe 0.872
    estimator.predictor.getEta shouldBe 0.99912
  }
}
