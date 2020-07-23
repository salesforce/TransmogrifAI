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
import org.apache.spark.sql.functions.{abs, avg}
import org.junit.runner.RunWith
import org.scalatest.{AppendedClues, Assertion, FunSpec}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpRegressionEvaluatorTest extends FunSpec with AppendedClues with TestSparkContext {

  import spark.implicits._

  describe(Spec[OpRegressionEvaluator]) {
    new SimpleEvaluationFixture {

      it("should copy") {
        val evaluator = newEvaluator()
        val theCopy = evaluator.copy(ParamMap())
        theCopy.uid shouldBe evaluator.uid
      }

    }

    describe("should evaluate the metrics from") {
      new EstimatorBasedEvaluationFixture {

        it("a model selector") {
          val model = testEstimatorSelector.fit(ds)
          val transformedData = model.setInput(label, features).transform(ds)
          val metrics = testEvaluatorSelector.evaluateAll(transformedData)

          metrics.RootMeanSquaredError should be <= 1E-12 withClue "rmse should be close to 0"
          metrics.MeanSquaredError should be <= 1E-24 withClue "mse should be close to 0"
          metrics.R2 shouldBe 1.0 withClue "R2 should equal 1.0"
          metrics.MeanAbsoluteError should be <= 1E-12 withClue "mae should be close to 0"
          checkMetricsNonEmpty(metrics)
        }

        it("a single model") {
          val model = testEstimator.fit(ds)
          val transformedData = model.setInput(label, features).transform(ds)
          val metrics = testEvaluator.evaluateAll(transformedData)

          metrics.RootMeanSquaredError should be <= 1E-12 withClue "rmse should be close to 0"
          metrics.MeanSquaredError should be <= 1E-24 withClue "mse should be close to 0"
          metrics.R2 shouldBe 1.0 withClue "R2 should equal 1.0"
          metrics.MeanAbsoluteError should be <= 1E-12 withClue "mae should be close to 0"
          checkMetricsNonEmpty(metrics)
        }
      }
    }

    describe(", in handling percentage error histogram,") {

      describe("in validating parameters,") {
        new SimpleEvaluationFixture {

          it("should fail on empty bins") {
            intercept[java.lang.IllegalArgumentException](newEvaluator()
              .setPercentageErrorHistogramBins(Array())
            )
          }

          it("should fail on unsorted bins") {
            intercept[java.lang.IllegalArgumentException](newEvaluator()
              .setPercentageErrorHistogramBins(Array(1.0, 0.0, 2.0))
            )
          }

          it("should fail on incorrect scaled cutoff values") {
            intercept[java.lang.IllegalArgumentException](newEvaluator()
              .setScaledErrorCutoff(-1.0))
          }

          it("should fail on incorrect values for the smart cutoff value ratio") {
            intercept[java.lang.IllegalArgumentException](newEvaluator()
              .setSmartCutoffRatio(-1.0))
          }

        }
      }

      describe("in setting parameters,") {
        new SimpleEvaluationFixture {

          it("should allow setting the histogram bins") {
            val metrics = newEvaluator()
              .setPercentageErrorHistogramBins(Array(-100.0, 0.0, 100.0))
              .evaluateAll(dataset)
            checkMetricsNonEmpty(metrics)
          }

          it("should allow setting the scaled cutoff value") {
            val metrics = newEvaluator()
              .setScaledErrorCutoff(1.0)
              .evaluateAll(dataset)
            checkMetricsNonEmpty(metrics)
          }

          it("should allow smartly setting the cutoff value") {
            val metrics = newEvaluator()
              .setSmartCutoff(true)
              .evaluateAll(dataset)
            checkMetricsNonEmpty(metrics)
          }

          it("should allow setting the ratio for the smart cutoff value calculation") {
            val metrics = newEvaluator()
              .setSmartCutoffRatio(1.0)
              .evaluateAll(dataset)
            checkMetricsNonEmpty(metrics)
          }

        }
      }

      describe("in calculating the histogram") {
        new RealisticEvaluationFixture {

          it("should handle the edge case where the data set is empty") {
            val bins = Array(Double.NegativeInfinity) ++ (-1.0 to 1.0 by 0.1) ++ Array(Double.PositiveInfinity)
            val metrics = newEvaluator()
              .setPercentageErrorHistogramBins(bins)
              .evaluateAll(spark.emptyDataset[EvalRow])
            metrics.SignedPercentageErrorHistogram.counts shouldBe Array.fill(bins.length - 1)(0L)
          }

          it("should have a number of total counts equal to the data point count") {
            val metrics = newEvaluator()
              .evaluateAll(dataset)
            metrics.SignedPercentageErrorHistogram.counts.sum shouldBe dataset.count()
          }

          it("should return the bins as set") {
            val bins = Array(Double.NegativeInfinity) ++ (-1.0 to 1.0 by 0.1) ++ Array(Double.PositiveInfinity)
            val metrics = newEvaluator()
              .setPercentageErrorHistogramBins(bins)
              .evaluateAll(dataset)
            metrics.SignedPercentageErrorHistogram.bins shouldBe bins
          }

          it("should result in N-1 counts for N bins") {
            val bins = Array(Double.NegativeInfinity) ++ (-1.0 to 1.0 by 0.1) ++ Array(Double.PositiveInfinity)
            val metrics = newEvaluator()
              .setPercentageErrorHistogramBins(bins)
              .evaluateAll(dataset)
            metrics.SignedPercentageErrorHistogram.counts.size shouldBe bins.length - 1
          }

          it("should do correct scaled error calculation with a custom cutoff") {
            val metrics = newEvaluator()
              .setPercentageErrorHistogramBins(
                Array(Double.NegativeInfinity, -500.0, -100.0, 0.0, 100.0, 500.0, Double.PositiveInfinity)
              )
              .setScaledErrorCutoff(10000.0) // Squeezes everything into center bins
              .evaluateAll(dataset)
            metrics.SignedPercentageErrorHistogram.counts shouldBe Array(0L, 0L, 4L, 6L, 0L, 0L)
          }

          it("should do correct smart cutoff value calculation and allow getting the result") {
            val evaluator = newEvaluator()
              .setSmartCutoff(true)
            evaluator.evaluateAll(dataset)
            val expected = Some(dataset.select(avg(abs($"label"))).map(_.getDouble(0)).first() * 0.1)
            evaluator.getScaledErrorCutoff shouldBe expected
          }

          it("should attempt correct smart cutoff value calculation even if all labels are 0") {
            val evaluator = newEvaluator()
              .setSmartCutoff(true)
            evaluator.evaluateAll(datasetWithZeroLabels)
            evaluator.getScaledErrorCutoff shouldBe Option(1E-3)
          }

          it("should do correct smart cutoff value calculation with a custom ratio") {
            val evaluator = newEvaluator()
              .setSmartCutoff(true)
              .setSmartCutoffRatio(0.2)
            evaluator.evaluateAll(dataset)
            val expected = Some(dataset.select(avg(abs($"label"))).map(_.getDouble(0)).first() * 0.2)
            evaluator.getScaledErrorCutoff shouldBe expected
          }

          it("should ignore data values outside the bins") {
            val metrics = newEvaluator()
              .setPercentageErrorHistogramBins(Array(-10.0, 0.0, 10.0))
              .evaluateAll(dataset)
            metrics.SignedPercentageErrorHistogram.counts shouldBe Array(1L, 3L)
          }

          it("should ignore NaNs in the labels and predictions") {
            val metrics = newEvaluator()
              .evaluateAll(datasetWithNaNs)
            metrics.SignedPercentageErrorHistogram.counts.sum shouldBe 3L
          }

          it("should calculate the correct histogram in case of zero or negative labels") {
            val bins = Array(Double.NegativeInfinity, -1000.0, -10.0, 0.0, 10.0, 1000.0, Double.PositiveInfinity)
            val metrics = newEvaluator()
              .setPercentageErrorHistogramBins(bins)
              .evaluateAll(dataset)
            metrics.SignedPercentageErrorHistogram.counts shouldBe Array(1L, 2L, 1L, 3L, 2L, 1L)
          }

          it("should calculate the correct histogram in case of perfect predictions") {
            val bins = Array(Double.NegativeInfinity, -100.0, 0.0, 100.0, Double.PositiveInfinity)
            val metrics = newEvaluator()
              .setPercentageErrorHistogramBins(bins)
              .evaluateAll(datasetPerfect)
            metrics.SignedPercentageErrorHistogram.counts shouldBe Array(0L, 0L, datasetPerfect.count().toLong, 0L)
          }

        }
      }
    }
  }

  // Helpers

  private def newEvaluator()(): OpRegressionEvaluator = {
    new OpRegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
  }

  private def checkMetricsNonEmpty(metrics: RegressionMetrics): Assertion = {
    metrics.SignedPercentageErrorHistogram.bins should not be empty
    metrics.SignedPercentageErrorHistogram.counts should not be empty
  }

  // Fixtures

  trait SimpleEvaluationFixture {
    val dataset = Seq(EvalRow(0.0, Map("prediction" -> 0.1))).toDS()
  }

  trait RealisticEvaluationFixture {
    val dataset = Seq(
      EvalRow(-10.0, Map("prediction" -> -11.0)),
      EvalRow(-4.0, Map("prediction" -> -8.0)),
      EvalRow(-2.0, Map("prediction" -> 0.1)),
      EvalRow(0.0, Map("prediction" -> -0.1)),
      EvalRow(0.0, Map("prediction" -> 0.0)),
      EvalRow(0.0, Map("prediction" -> 0.0)),
      EvalRow(0.0, Map("prediction" -> 0.1)),
      EvalRow(2.0, Map("prediction" -> 0.0)),
      EvalRow(4.0, Map("prediction" -> 4.0)),
      EvalRow(10.0, Map("prediction" -> 100.0))
    ).toDS()
    val datasetWithNaNs = Seq(
      EvalRow(-2.0, Map("prediction" -> 0.1)),
      EvalRow(Double.NaN, Map("prediction" -> 0.0)),
      EvalRow(0.0, Map("prediction" -> Double.NaN)),
      EvalRow(2.0, Map("prediction" -> 0.0)),
      EvalRow(2.0, Map("prediction" -> 0.0))
    ).toDS()
    val datasetWithZeroLabels = Seq(
      EvalRow(0.0, Map("prediction" -> 0.1)),
      EvalRow(0.0, Map("prediction" -> 0.0)),
      EvalRow(0.0, Map("prediction" -> 0.1)),
      EvalRow(0.0, Map("prediction" -> 0.0)),
      EvalRow(0.0, Map("prediction" -> 0.0))
    ).toDS()
    val datasetPerfect = Seq(
      EvalRow(1.0, Map("prediction" -> 1.0)),
      EvalRow(1.0, Map("prediction" -> 1.0)),
      EvalRow(1.0, Map("prediction" -> 1.0)),
      EvalRow(1.0, Map("prediction" -> 1.0)),
      EvalRow(1.0, Map("prediction" -> 1.0))
    ).toDS()
  }

  trait EstimatorBasedEvaluationFixture {
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
      ).
        map(v => v._1.toRealNN -> v._2.toOPVector)
    )
    val label = rawLabel.copy(isResponse = true)

    val testEstimator = new OpLinearRegression().setInput(label, features)
    val prediction = testEstimator.getOutput()
    val testEvaluator = new OpRegressionEvaluator().setLabelCol(label).setPredictionCol(prediction)

    val lr = new OpLogisticRegression()
    val testEstimatorSelector = RegressionModelSelector.withTrainValidationSplit(dataSplitter = None, trainRatio = 0.5,
      modelsAndParameters = Seq(lr -> new ParamGridBuilder().addGrid(lr.regParam, Array(0.0)).build()))
      .setInput(label, features)
    val predictionSelector = testEstimatorSelector.getOutput()
    val testEvaluatorSelector = new OpRegressionEvaluator().setLabelCol(label).setPredictionCol(predictionSelector)
  }

}

case class EvalRow(label: Double, prediction: Map[String, Double])
