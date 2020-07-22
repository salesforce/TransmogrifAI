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
import com.salesforce.op.stages.impl.classification.OpLogisticRegression
import com.salesforce.op.stages.impl.regression.{OpLinearRegression, RegressionModelSelector}
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.junit.runner.RunWith
import org.scalatest.{AppendedClues, Assertion, FunSpec}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpRegressionEvaluatorTest extends FunSpec with AppendedClues with TestSparkContext {

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

  val lr = new OpLogisticRegression()
  val lrParams = new ParamGridBuilder().addGrid(lr.regParam, Array(0.0)).build()

  val testEstimator = RegressionModelSelector.withTrainValidationSplit(dataSplitter = None, trainRatio = 0.5,
    modelsAndParameters = Seq(lr -> lrParams))
    .setInput(label, features)

  val prediction = testEstimator.getOutput()
  val testEvaluator = new OpRegressionEvaluator().setLabelCol(label).setPredictionCol(prediction)

  val testEstimator2 = new OpLinearRegression().setInput(label, features)

  val prediction2 = testEstimator2.getOutput()
  val testEvaluator2 = new OpRegressionEvaluator().setLabelCol(label).setPredictionCol(prediction2)


  describe(Spec[OpRegressionEvaluator]) {
    it("should copy") {
      val testEvaluatorCopy = testEvaluator.copy(ParamMap())
      testEvaluatorCopy.uid shouldBe testEvaluator.uid
    }

    it("should evaluate the metrics from a model selector") {
      val model = testEstimator.fit(ds)
      val transformedData = model.setInput(label, features).transform(ds)
      val metrics = testEvaluator.evaluateAll(transformedData)

      metrics.RootMeanSquaredError should be <= 1E-12 withClue "rmse should be close to 0"
      metrics.MeanSquaredError should be <= 1E-24 withClue "mse should be close to 0"
      metrics.R2 shouldBe 1.0 withClue "R2 should equal 1.0"
      metrics.MeanAbsoluteError should be <= 1E-12 withClue "mae should be close to 0"
      assertHistogramNotEmpty(metrics)
    }

    it("should evaluate the metrics from a single model") {
      val model = testEstimator2.fit(ds)
      val transformedData = model.setInput(label, features).transform(ds)
      val metrics = testEvaluator2.evaluateAll(transformedData)

      metrics.RootMeanSquaredError should be <= 1E-12 withClue "rmse should be close to 0"
      metrics.MeanSquaredError should be <= 1E-24 withClue "mse should be close to 0"
      metrics.R2 shouldBe 1.0 withClue "R2 should equal 1.0"
      metrics.MeanAbsoluteError should be <= 1E-12 withClue "mae should be close to 0"
      assertHistogramNotEmpty(metrics)
    }

    describe(", in handling percentage error histogram,") {

      it("should fail on empty bins") {
        intercept[java.lang.IllegalArgumentException](new OpRegressionEvaluator()
          .setRelativeErrorHistogramBins(Array())
        )
      }

      it("should fail on unsorted bins") {
        intercept[java.lang.IllegalArgumentException](new OpRegressionEvaluator()
          .setRelativeErrorHistogramBins(Array(1.0, 0.0, 2.0))
        )
      }

      it("should allow setting the histogram bins") {
        val data =
        val evaluator = new OpRegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .evaluateAll(data)
        val expectedBins = Array()
        val expectedCounts = Array()
      }

      it("should fail on incorrect scaled cutoff values") {
        intercept[java.lang.IllegalArgumentException](new OpRegressionEvaluator()
          .setScaledErrorCutoff(-1.0))
      }

      it("should allow setting the scaled cutoff value") {}

      it("should fail on incorrect values for the smart cutoff value ratio") {
        intercept[java.lang.IllegalArgumentException](new OpRegressionEvaluator()
          .setSmartCutoffRatio(-1.0))
      }

      it("should allow smartly setting the cutoff value") {}

      it("should allow setting the ratio for the smart cutoff value calculation") {}

      it("should ignore data values outside the bins") {}

      it("should have a number of total counts equal to the data point count") {}

      it("should ignore NaNs in the labels") {}

      it("should result in N-1 counts for N bins") {}

      it("should calculate the correct histogram") {}

      it("should calculate the correct histogram in case of zero or negative labels") {}

    }

  }

  private def assertHistogramNotEmpty(metrics: RegressionMetrics): Assertion = {
    metrics.relativeErrorHistogramBins should not be empty withClue "there should be histogram bins"
    metrics.relativeErrorHistogramCounts should not be empty withClue "there should be histogram counts"
  }

}
