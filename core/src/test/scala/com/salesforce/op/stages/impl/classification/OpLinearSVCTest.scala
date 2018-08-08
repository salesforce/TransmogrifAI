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
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpLinearSVCTest extends OpEstimatorSpec[Prediction, OpPredictorWrapperModel[LinearSVCModel],
  OpPredictorWrapper[LinearSVC, LinearSVCModel]] with PredictionEquality {

  override def specName: String = classOf[OpLinearSVC].getSimpleName

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
  val estimator = new OpLinearSVC().setInput(feature1, feature2)

  val expectedResult = Seq(
    Prediction(1.0, Vectors.dense(Array(-1.33, 1.33))),
    Prediction(0.0, Vectors.dense(Array(1.04, -1.04))),
    Prediction(0.0, Vectors.dense(Array(2.69, -2.69))),
    Prediction(1.0, Vectors.dense(Array(-1.32, 1.32))),
    Prediction(1.0, Vectors.dense(Array(-2.11, 2.11))),
    Prediction(0.0, Vectors.dense(Array(4.41, -4.41))),
    Prediction(1.0, Vectors.dense(Array(-1.46, 1.46))),
    Prediction(0.0, Vectors.dense(Array(1.42, -1.42)))
  )


  it should "allow the user to set the desired spark parameters" in {
    estimator
      .setRegParam(0.1)
      .setMaxIter(20)
      .setTol(1E-4)
    estimator.fit(inputData)

    estimator.predictor.getRegParam shouldBe 0.1
    estimator.predictor.getMaxIter shouldBe 20
    estimator.predictor.getTol shouldBe 1E-4
  }
}


