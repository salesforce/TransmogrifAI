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
    Prediction(1.0, Array(-5.172501101023487, 6.543830316806457), Array(8.159402805507398E-6, 0.9999918405971945)),
    Prediction(0.0, Array(7.708825172282052, -7.846086755046684), Array(0.999999824374527, 1.7562547311755836E-7)),
    Prediction(0.0, Array(6.958195281529266, -6.847797459689109), Array(0.999998990437764, 1.009562235990671E-6)),
    Prediction(1.0, Array(-5.142996733536394, 6.690315031103952), Array(7.258633113002052E-6, 0.9999927413668871)),
    Prediction(1.0, Array(-5.161407834451036, 6.693896966545731), Array(7.100737530622016E-6, 0.9999928992624694)),
    Prediction(0.0, Array(6.957344333140615, -6.846638851649445), Array(0.9999989884069539, 1.0115930460497824E-6)),
    Prediction(1.0, Array(-5.145799479536089, 6.690944181932334), Array(7.233765109863128E-6, 0.9999927662348902)),
    Prediction(0.0, Array(7.548936676180427, -7.735803331602069), Array(0.9999997698973303, 2.3010266964026535E-7))
  )

  it should "allow the user to set the desired spark parameters" in {
    estimator.setMaxIter(50).setBlockSize(2).setSeed(42)
    estimator.fit(inputData)
    estimator.predictor.getMaxIter shouldBe 50
    estimator.predictor.getBlockSize shouldBe 2
    estimator.predictor.getSeed shouldBe 42
  }
}


