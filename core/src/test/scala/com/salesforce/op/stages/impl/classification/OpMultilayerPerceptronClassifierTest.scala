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

  override def specName: String = classOf[OpMultilayerPerceptronClassifier].getSimpleName

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

  val expectedResult = Seq(
    Prediction(1.0, Array(-9.655814651428148, 9.202335441336952), Array(6.456683124562021E-9, 0.9999999935433168)),
    Prediction(0.0, Array(9.475612761543069, -10.617525149157993), Array(0.9999999981221492, 1.877850786773977E-9)),
    Prediction(0.0, Array(9.715293827870028, -10.885255922155942), Array(0.9999999988694366, 1.130563392364822E-9)),
    Prediction(1.0, Array(-9.66776357765489, 9.215079716735316), Array(6.299199338896916E-9, 0.9999999937008006)),
    Prediction(1.0, Array(-9.668041712561456, 9.215387575592239), Array(6.2955091287182745E-9, 0.9999999937044908)),
    Prediction(0.0, Array(9.692904797559496, -10.860273756796797), Array(0.9999999988145918, 1.1854083109077814E-9)),
    Prediction(1.0, Array(-9.667687253240183, 9.214995747770411), Array(6.300209139771467E-9, 0.9999999936997908)),
    Prediction(0.0, Array(9.703097414537668, -10.872171694864653), Array(0.9999999988404908, 1.1595091005698914E-9))
  )

  it should "allow the user to set the desired spark parameters" in {
    estimator.setMaxIter(50).setBlockSize(2).setSeed(42)
    estimator.fit(inputData)
    estimator.predictor.getMaxIter shouldBe 50
    estimator.predictor.getBlockSize shouldBe 2
    estimator.predictor.getSeed shouldBe 42
  }
}


