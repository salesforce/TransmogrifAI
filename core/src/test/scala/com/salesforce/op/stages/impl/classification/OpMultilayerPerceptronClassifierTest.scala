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
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpMultilayerPerceptronClassifierTest extends OpEstimatorSpec[Prediction,
  OpPredictorWrapperModel[MultilayerPerceptronClassificationModel],
  OpPredictorWrapper[MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel]] with PredictionEquality {

  override def specName: String = Spec[OpMultilayerPerceptronClassifier]

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
  val estimator = new OpMultilayerPerceptronClassifier()
    .setInput(feature1, feature2)
    .setLayers(Array(3, 5, 4, 2))
    .setSeed(42)

  val expectedResult = Seq(
    Prediction(1.0, Array(-15.925965267326575, 19.709874206655577), Array(3.3385013674725553E-16, 0.9999999999999996)),
    Prediction(0.0, Array(10.46805725397906, -7.984143456049299), Array(0.999999990310284, 9.68971600310748E-9)),
    Prediction(0.0, Array(10.623898149483312, -8.149926623454933), Array(0.9999999929752399, 7.024760042487556E-9)),
    Prediction(1.0, Array(-16.96293489394458, 20.825501963956178), Array(3.878737534561395E-17, 1.0)),
    Prediction(1.0, Array(-16.949682428916343, 20.81125559615973), Array(3.9868783482134044E-17, 1.0)),
    Prediction(0.0, Array(10.67843222379218, -8.207747433881352), Array(0.999999993721782, 6.2782179418840885E-9)),
    Prediction(1.0, Array(-16.958513812076358, 20.820756918667733), Array(3.914453976667534E-17, 1.0)),
    Prediction(0.0, Array(10.398506602006975, -7.914192708632671), Array(0.9999999888597294, 1.1140270489918198E-8))
  )

  it should "allow the user to set the desired spark parameters" in {
    estimator.setMaxIter(50).setBlockSize(2).setSeed(42)
    estimator.fit(inputData)
    estimator.predictor.getMaxIter shouldBe 50
    estimator.predictor.getBlockSize shouldBe 2
    estimator.predictor.getSeed shouldBe 42
  }
}


