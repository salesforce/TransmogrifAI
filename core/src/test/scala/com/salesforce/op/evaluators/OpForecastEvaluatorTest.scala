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
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpForecastEvaluatorTest extends FlatSpec with TestSparkContext {

  import spark.implicits._

  val (ds, rawLabel, features) = TestFeatureBuilder[RealNN, OPVector](
    Seq(
      (12.0, Vectors.dense(1.0, 4.3, 1.3)),
      (22.0, Vectors.dense(2.0, 0.3, 0.1)),
      (32.0, Vectors.dense(3.0, 3.9, 4.3)),
      (42.0, Vectors.dense(4.0, 1.3, 0.9)),
      (52.0, Vectors.dense(5.0, 4.7, 1.3)),
      (17.0, Vectors.dense(1.0, 4.3, 1.3)),
      (27.0, Vectors.dense(2.0, 0.3, 0.1)),
      (37.0, Vectors.dense(3.0, 3.9, 4.3)),
      (47.0, Vectors.dense(4.0, 1.3, 0.9)),
      (57.0, Vectors.dense(5.0, 4.7, 1.3))
    ).map(v => v._1.toRealNN -> v._2.toOPVector)
  )

  val label = rawLabel.copy(isResponse = true)

  val lr = new OpLogisticRegression()
  val lrParams = new ParamGridBuilder().addGrid(lr.regParam, Array(0.0)).build()

  val testEstimator = RegressionModelSelector.withTrainValidationSplit(dataSplitter = None, trainRatio = .5,
    modelsAndParameters = Seq(lr -> lrParams), seed = 1239871928731L)
    .setInput(label, features)

  val prediction = testEstimator.getOutput()
  val testEvaluator = new OpForecastEvaluator().setLabelCol(label).setPredictionCol(prediction)

  val testEstimator2 = new OpLinearRegression().setInput(label, features)

  val prediction2 = testEstimator2.getOutput()
  val testEvaluator2 = new OpForecastEvaluator().setLabelCol(label).setPredictionCol(prediction2)


  Spec[OpForecastEvaluator] should "copy" in {
    val testEvaluatorCopy = testEvaluator.copy(ParamMap())
    testEvaluatorCopy.uid shouldBe testEvaluator.uid
  }

  it should "evaluate the metrics from a model selector" in {
    val model = testEstimator.fit(ds)
    val transformedData = model.setInput(label, features).transform(ds)
    val metrics = testEvaluator.evaluateAll(transformedData).toMetadata()

    metrics.getDouble(ForecastEvalMetrics.SMAPE.toString) shouldBe (0.08979 +- 1e-4)

  }

  it should "evaluate the metrics from a single model" in {
    val model = testEstimator2.fit(ds)
    val transformedData = model.setInput(label, features).transform(ds)
    val metrics = testEvaluator2.evaluateAll(transformedData).toMetadata()
    metrics.getDouble(ForecastEvalMetrics.SMAPE.toString) shouldBe (0.09013 +- 1e-4)
  }

  it should "evaluate the metrics when data is empty" in {
    val data = Seq().map(x => (x, Map("prediction" -> x)))
    val df = spark.sparkContext.parallelize(data).toDF("f1", "r1")
    val metrics = new OpForecastEvaluator().setLabelCol("f1").setPredictionCol("r1").evaluateAll(df).toMetadata()
    metrics.getDouble(ForecastEvalMetrics.SMAPE.toString).isNaN shouldBe true
  }


  it should "verify that SeasonalAbsDiff works" in {
    def testSimple(stop: Int = 20, window: Int = 1, expected: Double): Unit = {
      val l = (1 to stop).map(x => SeasonalAbsDiff.apply(window, x)).fold(SeasonalAbsDiff.zero(window))(_.combine(_))
      if (expected.isNaN) {
        l.seasonalError(stop).isNaN shouldBe true
      } else {
        l.seasonalError(stop) shouldBe expected
      }
    }

    testSimple(20, 1, 1.0)
    testSimple(20, 5, 5.0)
    testSimple(20, 20, Double.NaN)


    def testSignal(dataLength: Int, windowLength: Int): Unit = {
      val signal = (1 to dataLength).map(x => Math.sin((x / dataLength) * 2.0 * Math.PI))
      val expected = (1 to (dataLength - windowLength)).map(x => {
        Math.abs(
          Math.sin((x / dataLength) * 2.0 * Math.PI)
            - Math.sin((windowLength / dataLength) * 2.0 * Math.PI + (x / dataLength) * 2.0 * Math.PI)
        )
      }).sum / (dataLength - windowLength)


      val res = signal
        .map(x => SeasonalAbsDiff.apply(windowLength, x))
        .fold(SeasonalAbsDiff.zero(windowLength))(_.combine(_))
        .seasonalError(dataLength)
      res shouldEqual (expected +- 1e-6)
    }

    (1 to 50).foreach(x => testSignal(100, x))

  }

  it should "verify that Forecast Metrics works" in {
    val dataLength = 100
    val seasonalWindow = 25
    val data = (0 until dataLength).map(x => {
      val y = Math.sin((x.toDouble / dataLength) * 2.0 * Math.PI)
      (y, Map("prediction" -> y * 1.2))
    })

    val df = spark.sparkContext.parallelize(data).toDF("f1", "r1")
    val metrics = new OpForecastEvaluator(seasonalWindow)
      .setLabelCol("f1").setPredictionCol("r1").evaluateAll(df).toMetadata()

    metrics.getDouble(ForecastEvalMetrics.MASE.toString) shouldBe (0.16395 +- 1e-5)
    metrics.getDouble(ForecastEvalMetrics.SeasonalError.toString) shouldBe (0.77634 +- 1e-5)
    metrics.getDouble(ForecastEvalMetrics.SMAPE.toString) shouldBe (0.18 +- 1e-3)
  }

}
