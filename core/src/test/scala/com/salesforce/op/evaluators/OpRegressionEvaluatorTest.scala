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

package com.salesforce.op.evaluators

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.regression.{OpLinearRegression, RegressionModelSelector}
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry.LinearRegression
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpRegressionEvaluatorTest extends FlatSpec with TestSparkContext {

  val (ds, rawLabel, features) = TestFeatureBuilder[RealNN, OPVector](
    Seq(
      (10.0, Vectors.dense(1.0, 4.3, 1.3)),
      (20.0, Vectors.dense(2.0, 0.3, 0.1)),
      (30.0, Vectors.dense(3.0, 3.9, 4.3)),
      (40.0, Vectors.dense(4.0, 1.3, 0.9)),
      (50.0, Vectors.dense(5.0, 4.7, 1.3)),
      (10.0, Vectors.dense(1.0, 4.3, 1.3)),
      (20.0, Vectors.dense(2.0, 0.3, 0.1)),
      (30.0, Vectors.dense(3.0, 3.9, 4.3)),
      (40.0, Vectors.dense(4.0, 1.3, 0.9)),
      (50.0, Vectors.dense(5.0, 4.7, 1.3))
    ).map(v => v._1.toRealNN -> v._2.toOPVector)
  )

  val label = rawLabel.copy(isResponse = true)

  val testEstimator = RegressionModelSelector.withTrainValidationSplit(dataSplitter = None, trainRatio = 0.5)
    .setModelsToTry(LinearRegression)
    .setLinearRegressionRegParam(0)
    .setInput(label, features)

  val prediction = testEstimator.getOutput()
  val testEvaluator = new OpRegressionEvaluator().setLabelCol(label).setPredictionCol(prediction)

  val testEstimator2 = new OpLinearRegression().setInput(label, features)

  val prediction2 = testEstimator2.getOutput()
  val testEvaluator2 = new OpRegressionEvaluator().setLabelCol(label).setFullPredictionCol(prediction2)


  Spec[OpRegressionEvaluator] should "copy" in {
    val testEvaluatorCopy = testEvaluator.copy(ParamMap())
    testEvaluatorCopy.uid shouldBe testEvaluator.uid
  }

  it should "evaluate the metrics when there are 3 outputs from the model" in {
    val model = testEstimator.fit(ds)
    val transformedData = model.setInput(label, features).transform(ds)
    val metrics = testEvaluator.evaluateAll(transformedData).toMetadata

    assert(metrics.getDouble(RegressionEvalMetrics.RootMeanSquaredError.toString) <= 1E-12, "rmse should be close to 0")
    assert(metrics.getDouble(RegressionEvalMetrics.MeanSquaredError.toString) <= 1E-24, "mse should be close to 0")
    assert(metrics.getDouble(RegressionEvalMetrics.R2.toString) == 1.0, "R2 should equal 1.0")
    assert(metrics.getDouble(RegressionEvalMetrics.MeanAbsoluteError.toString) <= 1E-12, "mae should be close to 0")
  }

  it should "evaluate the metrics when there is a prediction output from the model" in {
    val model = testEstimator2.fit(ds)
    val transformedData = model.setInput(label, features).transform(ds)
    val metrics = testEvaluator2.evaluateAll(transformedData).toMetadata

    assert(metrics.getDouble(RegressionEvalMetrics.RootMeanSquaredError.toString) <= 1E-12, "rmse should be close to 0")
    assert(metrics.getDouble(RegressionEvalMetrics.MeanSquaredError.toString) <= 1E-24, "mse should be close to 0")
    assert(metrics.getDouble(RegressionEvalMetrics.R2.toString) == 1.0, "R2 should equal 1.0")
    assert(metrics.getDouble(RegressionEvalMetrics.MeanAbsoluteError.toString) <= 1E-12, "mae should be close to 0")
  }
}
