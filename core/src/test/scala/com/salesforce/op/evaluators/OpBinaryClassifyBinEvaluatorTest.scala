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

import com.salesforce.op.test.TestSparkContext
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpBinaryClassifyBinEvaluatorTest extends FlatSpec with TestSparkContext {

  val labelName = "label"
  val predictionLabel = "pred"

  val dataset_test = Seq(
    (Map("probability_1" -> 0.99999, "probability_0" -> 0.0001, "prediction" -> 1.0), 1.0),
    (Map("probability_1" -> 0.99999, "probability_0" -> 0.0001, "prediction" -> 1.0), 1.0),
    (Map("probability_1" -> 0.00541, "probability_0" -> 0.99560, "prediction" -> 1.0), 0.0),
    (Map("probability_1" -> 0.70, "probability_0" -> 0.30, "prediction" -> 1.0), 0.0),
    (Map("probability_1" -> 0.001, "probability_0" -> 0.999, "prediction" -> 0.0), 0.0)
  )

  val dataset_skewed = Seq(
    (Map("probability_1" -> 0.99999, "probability_0" -> 0.0001, "prediction" -> 1.0), 1.0),
    (Map("probability_1" -> 0.99999, "probability_0" -> 0.0001, "prediction" -> 1.0), 1.0),
    (Map("probability_1" -> 0.9987, "probability_0" -> 0.001, "prediction" -> 1.0), 1.0),
    (Map("probability_1" -> 0.946, "probability_0" -> 0.0541, "prediction" -> 1.0), 1.0)
  )

  val emptyDataSet = Seq.empty[(Map[String, Double], Double)]

  Spec[OpBinaryClassifyBinEvaluator] should "return the bin metrics" in {
    val df = spark.createDataFrame(dataset_test).toDF(predictionLabel, labelName)

    val metrics = new OpBinaryClassifyBinEvaluator(numBins = 4)
      .setLabelCol(labelName).setPredictionCol(predictionLabel).evaluateAll(df)

    BigDecimal(metrics.BrierScore).setScale(3, BigDecimal.RoundingMode.HALF_UP).toDouble shouldBe 0.098
    metrics.BinCenters shouldBe Seq(0.125, 0.375, 0.625, 0.875)
    metrics.NumberOfDataPoints shouldBe Seq(2, 0, 1, 2)
    metrics.AverageScore shouldBe Seq(0.003205, 0.0, 0.7, 0.99999)
    metrics.AverageConversionRate shouldBe Seq(0.0, 0.0, 0.0, 1.0)
  }

  it should "return the empty bin metrics for numBins == 0" in {
    val df = spark.createDataFrame(dataset_test).toDF(predictionLabel, labelName)

    val metrics = new OpBinaryClassifyBinEvaluator(numBins = 0)
      .setLabelCol(labelName).setPredictionCol(predictionLabel).evaluateAll(df)

    metrics.BrierScore shouldBe 0.0
    metrics.BinCenters shouldBe Seq.empty[Double]
    metrics.NumberOfDataPoints shouldBe Seq.empty[Long]
    metrics.AverageScore shouldBe Seq.empty[Double]
    metrics.AverageConversionRate shouldBe Seq.empty[Double]
  }

  it should "return the empty bin metrics for empty data" in {
    val df = spark.createDataFrame(emptyDataSet).toDF(predictionLabel, labelName)

    val metrics = new OpBinaryClassifyBinEvaluator(numBins = 10)
      .setLabelCol(labelName).setPredictionCol(predictionLabel).evaluateAll(df)

    metrics.BrierScore shouldBe 0.0
    metrics.BinCenters shouldBe Seq.empty[Double]
    metrics.NumberOfDataPoints shouldBe Seq.empty[Long]
    metrics.AverageScore shouldBe Seq.empty[Double]
    metrics.AverageConversionRate shouldBe Seq.empty[Double]
  }

  it should "return the bin metrics for skewed data" in {
    val df = spark.createDataFrame(dataset_skewed).toDF(predictionLabel, labelName)

    val metrics = new OpBinaryClassifyBinEvaluator(numBins = 5)
      .setLabelCol(labelName).setPredictionCol(predictionLabel).evaluateAll(df)

    metrics.BrierScore shouldBe 7.294225500000013E-4
    metrics.BinCenters shouldBe Seq(0.1, 0.3, 0.5, 0.7, 0.9)
    metrics.NumberOfDataPoints shouldBe Seq(0, 0, 0, 0, 4)
    metrics.AverageScore shouldBe Seq(0.0, 0.0, 0.0, 0.0, 0.98617)
    metrics.AverageConversionRate shouldBe Seq(0.0, 0.0, 0.0, 0.0, 1.0)
  }

  it should "return the default metric as BrierScore" in {
    val df = spark.createDataFrame(dataset_test).toDF(predictionLabel, labelName)

    val evaluator = new OpBinaryClassifyBinEvaluator(numBins = 4)
      .setLabelCol(labelName).setPredictionCol(predictionLabel)

    val brierScore = evaluator.getDefaultMetric(evaluator.evaluateAll(df))

    BigDecimal(brierScore).setScale(3, BigDecimal.RoundingMode.HALF_UP).toDouble shouldBe 0.098
  }
}
