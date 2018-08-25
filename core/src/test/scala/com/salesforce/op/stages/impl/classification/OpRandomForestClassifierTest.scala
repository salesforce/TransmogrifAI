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
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpRandomForestClassifierTest extends
  OpEstimatorSpec[Prediction, OpPredictorWrapperModel[RandomForestClassificationModel],
    OpPredictorWrapper[RandomForestClassifier, RandomForestClassificationModel]] with PredictionEquality {

  override def specName: String = Spec[OpRandomForestClassifier]

  lazy val (inputData, rawLabelMulti, featuresMulti) =
    TestFeatureBuilder[RealNN, OPVector]("labelMulti", "featuresMulti",
      Seq(
        (1.0.toRealNN, Vectors.dense(12.0, 4.3, 1.3).toOPVector),
        (0.0.toRealNN, Vectors.dense(0.0, 0.3, 0.1).toOPVector),
        (2.0.toRealNN, Vectors.dense(1.0, 3.9, 4.3).toOPVector),
        (2.0.toRealNN, Vectors.dense(10.0, 1.3, 0.9).toOPVector),
        (1.0.toRealNN, Vectors.dense(15.0, 4.7, 1.3).toOPVector),
        (0.0.toRealNN, Vectors.dense(0.5, 0.9, 10.1).toOPVector),
        (1.0.toRealNN, Vectors.dense(11.5, 2.3, 1.3).toOPVector),
        (0.0.toRealNN, Vectors.dense(0.1, 3.3, 0.1).toOPVector),
        (2.0.toRealNN, Vectors.dense(1.0, 4.0, 4.5).toOPVector),
        (2.0.toRealNN, Vectors.dense(10.0, 1.5, 1.0).toOPVector)
      )
    )

  val labelMulti = rawLabelMulti.copy(isResponse = true)

  val estimator = new OpRandomForestClassifier().setInput(labelMulti, featuresMulti)

  val expectedResult = Seq(
    Prediction(1.0, Array(0.0, 17.0, 3.0), Array(0.0, 0.85, 0.15)),
    Prediction(0.0, Array(19.0, 0.0, 1.0), Array(0.95, 0.0, 0.05)),
    Prediction(2.0, Array(0.0, 1.0, 19.0), Array(0.0, 0.05, 0.95)),
    Prediction(2.0, Array(1.0, 2.0, 17.0), Array(0.05, 0.1, 0.85)),
    Prediction(1.0, Array(0.0, 17.0, 3.0), Array(0.0, 0.85, 0.15)),
    Prediction(0.0, Array(16.0, 0.0, 4.0), Array(0.8, 0.0, 0.2)),
    Prediction(1.0, Array(1.0, 17.0, 2.0), Array(0.05, 0.85, 0.1)),
    Prediction(0.0, Array(17.0, 0.0, 3.0), Array(0.85, 0.0, 0.15)),
    Prediction(2.0, Array(2.0, 1.0, 17.0), Array(0.1, 0.05, 0.85)),
    Prediction(2.0, Array(1.0, 2.0, 17.0), Array(0.05, 0.1, 0.85))
  )

  it should "allow the user to set the desired spark parameters" in {
    estimator
      .setMaxDepth(10)
      .setImpurity(Impurity.Gini.sparkName)
      .setMaxBins(33)
      .setMinInstancesPerNode(2)
      .setMinInfoGain(0.2)
      .setSubsamplingRate(0.9)
      .setNumTrees(21)
      .setSeed(2L)
    estimator.fit(inputData)

    estimator.predictor.getMaxDepth shouldBe 10
    estimator.predictor.getMaxBins shouldBe 33
    estimator.predictor.getImpurity shouldBe Impurity.Gini.sparkName
    estimator.predictor.getMinInstancesPerNode shouldBe 2
    estimator.predictor.getMinInfoGain shouldBe 0.2
    estimator.predictor.getSubsamplingRate shouldBe 0.9
    estimator.predictor.getNumTrees shouldBe 21
    estimator.predictor.getSeed shouldBe 2L
  }

}
