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
    Prediction(1.0, Array(-8.539364696257962, 10.67130898750246), Array(4.5384799746525405E-9, 0.99999999546152)),
    Prediction(0.0, Array(10.590179532009554, -10.476815586211686), Array(0.999999999290879, 7.091208738628559E-10)),
    Prediction(0.0, Array(9.513859092221331, -9.401215393289661), Array(0.9999999939005941, 6.099405731305196E-9)),
    Prediction(1.0, Array(-8.542581739573867, 10.67512003391953), Array(4.506694955100369E-9, 0.999999995493305)),
    Prediction(1.0, Array(-8.54251860116924, 10.675044086443743), Array(4.507321816325889E-9, 0.9999999954926782)),
    Prediction(0.0, Array(9.677891306803922, -9.568722801536905), Array(0.9999999956217385, 4.378261484412989E-9)),
    Prediction(1.0, Array(-8.542523119151225, 10.675049530892785), Array(4.507276912667043E-9, 0.999999995492723)),
    Prediction(0.0, Array(9.681761128645391, -9.57265451015669), Array(0.9999999956557628, 4.344237237393638E-9))
  )

  it should "allow the user to set the desired spark parameters" in {
    estimator.setMaxIter(50).setBlockSize(2).setSeed(42)
    estimator.fit(inputData)
    estimator.predictor.getMaxIter shouldBe 50
    estimator.predictor.getBlockSize shouldBe 2
    estimator.predictor.getSeed shouldBe 42
  }
}


