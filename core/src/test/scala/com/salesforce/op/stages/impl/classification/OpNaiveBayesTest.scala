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

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.PredictionEquality
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpPredictorWrapperModel}
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpNaiveBayesTest extends OpEstimatorSpec[Prediction, OpPredictorWrapperModel[NaiveBayesModel],
  OpPredictorWrapper[NaiveBayes, NaiveBayesModel]] with PredictionEquality {

  val (inputData, rawFeature1, feature2) = TestFeatureBuilder("label", "features",
    Seq[(RealNN, OPVector)](
      1.0.toRealNN -> Vectors.dense(12.0, 4.3, 1.3).toOPVector,
      0.0.toRealNN -> Vectors.dense(0.0, 0.3, 0.1).toOPVector,
      0.0.toRealNN -> Vectors.dense(1.0, 3.9, 4.3).toOPVector,
      1.0.toRealNN -> Vectors.dense(10.0, 1.3, 0.9).toOPVector,
      1.0.toRealNN -> Vectors.dense(15.0, 4.7, 1.3).toOPVector,
      0.0.toRealNN -> Vectors.dense(0.5, 0.9, 10.1).toOPVector,
      1.0.toRealNN -> Vectors.dense(11.5, 2.3, 1.3).toOPVector,
      0.0.toRealNN -> Vectors.dense(0.1, 3.3, 0.1).toOPVector
    )
  )
  val feature1 = rawFeature1.copy(isResponse = true)
  val estimator = new OpNaiveBayes().setInput(feature1, feature2)

  val expectedResult = Seq(
    Prediction(1.0, Array(-34.41, -14.85), Array(0.0, 1.0)),
    Prediction(0.0, Array(-1.07, -1.42), Array(0.58, 0.41)),
    Prediction(0.0, Array(-9.70, -17.99), Array(1.0, 0.0)),
    Prediction(1.0, Array(-26.22, -8.33), Array(0.0, 1.0)),
    Prediction(1.0, Array(-41.93, -16.49), Array(0.0, 1.0)),
    Prediction(0.0, Array(-8.60, -27.31), Array(1.0, 0.0)),
    Prediction(1.0, Array(-31.07, -11.44), Array(0.0, 1.0)),
    Prediction(0.0, Array(-4.54, -6.32), Array(0.85, 0.14))
  )


  it should "allow the user to set the desired spark parameters" in {
    estimator
      .setSmoothing(2)
    estimator.fit(inputData)

    estimator.predictor.getSmoothing shouldBe 2
  }
}


