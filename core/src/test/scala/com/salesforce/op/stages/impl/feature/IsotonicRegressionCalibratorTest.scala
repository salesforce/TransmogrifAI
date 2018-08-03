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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.features.{Feature, FeatureLike}
import com.salesforce.op.stages.impl.regression.IsotonicRegressionCalibrator
import com.salesforce.op.stages.sparkwrappers.specific.OpBinaryEstimatorWrapper
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.regression.{IsotonicRegression, IsotonicRegressionModel}
import org.apache.spark.sql._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class IsotonicRegressionCalibratorTest extends FlatSpec with TestSparkContext {

  val isoExpectedPredictions = Array(1, 2, 2, 2, 6, 16.5, 16.5, 17, 18)
  val isoExpectedModelBoundaries = Array(0, 1, 3, 4, 5, 6, 7, 8)
  val isoExpectedModelPredictions = Array(1, 2, 2, 6, 16.5, 16.5, 17.0, 18.0)

  val isoDataLabels = Seq(1, 2, 3, 1, 6, 17, 16, 17, 18)
  val isoTestData = isoDataLabels.zipWithIndex.map {
    case (label, i) => label.toRealNN -> i.toRealNN
  }

  val (isoScoresDF, isoLabels, isoScores): (DataFrame, Feature[RealNN], Feature[RealNN]) =
    TestFeatureBuilder(isoTestData)

  val antiExpectedPredictions = Array(7.0, 5.0, 4.0, 4.0, 1.0)
  val antiExpectedModelBoundaries = Array(0, 1, 2, 3, 4)
  val antiExpectedModelPredictions = Array(7.0, 5.0, 4.0, 4.0, 1.0)

  val antiDataLabels = Seq(7, 5, 3, 5, 1)
  val antiTestData = antiDataLabels.zipWithIndex.map {
    case (label, i) => label.toRealNN -> i.toRealNN
  }

  val (antiScoresDF, antiLabels, antiScores): (DataFrame, Feature[RealNN], Feature[RealNN]) =
    TestFeatureBuilder(antiTestData)

  Spec[IsotonicRegressionCalibrator] should "isotonically calibrate scores using shortcut" in {
    val calibratedScores = isoScores.toIsotonicCalibrated(isoLabels)

    val estimator = calibratedScores.originStage
      .asInstanceOf[OpBinaryEstimatorWrapper[RealNN, RealNN, RealNN, IsotonicRegression, IsotonicRegressionModel]]

    val model = estimator.fit(isoScoresDF).getSparkMlStage().get

    val predictionsDF = model.asInstanceOf[Transformer]
      .transform(isoScoresDF)

    validateOutput(calibratedScores, model, predictionsDF, true, isoExpectedPredictions, isoExpectedModelBoundaries,
      isoExpectedModelPredictions)
  }

  it should "isotonically calibrate scores" in {
    val isotonicCalibrator = new IsotonicRegressionCalibrator().setInput(isoLabels, isoScores)

    val calibratedScores = isotonicCalibrator.getOutput()

    val model = isotonicCalibrator.fit(isoScoresDF).getSparkMlStage().get

    val predictionsDF = model.asInstanceOf[Transformer]
      .transform(isoScoresDF)

    validateOutput(calibratedScores, model, predictionsDF, true, isoExpectedPredictions, isoExpectedModelBoundaries,
      isoExpectedModelPredictions)
  }

  it should "antitonically calibrate scores" in {
    val isIsotonic: Boolean = false
    val isotonicCalibrator = new IsotonicRegressionCalibrator().setIsotonic(isIsotonic).setInput(isoLabels, isoScores)

    val calibratedScores = isotonicCalibrator.getOutput()

    val model = isotonicCalibrator.fit(antiScoresDF).getSparkMlStage().get

    val predictionsDF = model.asInstanceOf[Transformer]
      .transform(antiScoresDF)

    validateOutput(calibratedScores, model, predictionsDF, isIsotonic, antiExpectedPredictions,
      antiExpectedModelBoundaries, antiExpectedModelPredictions)
  }

  def validateOutput(calibratedScores: FeatureLike[RealNN],
    model: IsotonicRegressionModel, predictionsDF: DataFrame, expectedIsIsotonic: Boolean,
    expectedPredictions: Array[Double], expectedModelBoundaries: Array[Int],
    expectedModelPredictions: Array[Double]): Unit = {

    val predictions = predictionsDF.select(calibratedScores.name).rdd.map { case Row(pred) => pred }.collect()
    val isIsotonic = model.getIsotonic

    isIsotonic should be(expectedIsIsotonic)
    predictions should contain theSameElementsInOrderAs expectedPredictions
    model.boundaries.toArray should contain theSameElementsInOrderAs expectedModelBoundaries
    model.predictions.toArray should contain theSameElementsInOrderAs expectedModelPredictions
  }
}
