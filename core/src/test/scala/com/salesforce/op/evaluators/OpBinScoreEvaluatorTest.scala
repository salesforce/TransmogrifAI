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

import com.salesforce.op.features.types.Prediction
import com.salesforce.op.test.TestSparkContext
import org.scalatest.FlatSpec
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpBinScoreEvaluatorTest extends FlatSpec with TestSparkContext {

  val labelName = "label"
  val predictionLabel = "pred"

  val dataset_test = Seq(
    (Prediction(1.0, Array(10.0, 10.0), Array(0.0001, 0.99999)), 1.0),
    (Prediction(1.0, Array(10.0, 10.0), Array(0.0001, 0.99999)), 1.0),
    (Prediction(1.0, Array(10.0, 10.0), Array(0.99560, 0.00541)), 0.0),
    (Prediction(1.0, Array(10.0, 10.0), Array(0.30, 0.70)), 0.0),
    (Prediction(0.0, Array(10.0, 10.0), Array(0.999, 0.001)), 0.0)
  ).map(v => (v._1.toDoubleMap, v._2))

  val dataset_test_df = spark.createDataFrame(dataset_test).toDF(predictionLabel, labelName)

  val dataset_skewed = Seq(
    (Prediction(1.0, Array(10.0, 10.0), Array(0.0001, 0.99999)), 1.0),
    (Prediction(1.0, Array(10.0, 10.0), Array(0.0001, 0.99999)), 1.0),
    (Prediction(1.0, Array(10.0, 10.0), Array(0.001, 0.9987)), 1.0),
    (Prediction(1.0, Array(10.0, 10.0), Array(0.0541, 0.946)), 1.0)
  ).map(v => (v._1.toDoubleMap, v._2))

  val dataset_skewed_df = spark.createDataFrame(dataset_skewed).toDF(predictionLabel, labelName)

  val emptyDataSet_df = spark.createDataFrame(Seq.empty[(Map[String, Double], Double)])
    .toDF(predictionLabel, labelName)

  Spec[OpBinScoreEvaluator] should "return the bin metrics" in {
    val metrics = new OpBinScoreEvaluator(numBins = 4)
      .setLabelCol(labelName).setPredictionCol(predictionLabel).evaluateAll(dataset_test_df)

    metrics shouldBe BinaryClassificationBinMetrics(
      0.09800605366,
      Seq(0.125, 0.375, 0.625, 0.875),
      Seq(2, 0, 1, 2),
      Seq(0.003205, 0.0, 0.7, 0.99999),
      Seq(0.0, 0.0, 0.0, 1.0))
  }

  it should "error on invalid number of bins" in {
    assertThrows[IllegalArgumentException] {
      new OpBinScoreEvaluator(numBins = 0)
        .setLabelCol(labelName).setPredictionCol(predictionLabel).evaluateAll(dataset_test_df)
    }
  }

  it should "evaluate the empty data" in {
    val metrics = new OpBinScoreEvaluator(numBins = 10)
      .setLabelCol(labelName).setPredictionCol(predictionLabel).evaluateAll(emptyDataSet_df)

    metrics shouldBe BinaryClassificationBinMetrics(0.0, Seq(), Seq(), Seq(), Seq())
  }

  it should "evaluate bin metrics for skewed data" in {
    val metrics = new OpBinScoreEvaluator(numBins = 5)
      .setLabelCol(labelName).setPredictionCol(predictionLabel).evaluateAll(dataset_skewed_df)

    metrics shouldBe BinaryClassificationBinMetrics(
      7.294225500000013E-4,
      Seq(0.1, 0.3, 0.5, 0.7, 0.9),
      Seq(0, 0, 0, 0, 4),
      Seq(0.0, 0.0, 0.0, 0.0, 0.98617),
      Seq(0.0, 0.0, 0.0, 0.0, 1.0))
  }

  it should "evaluate the default metric as BrierScore" in {
    val evaluator = new OpBinScoreEvaluator(numBins = 4)
      .setLabelCol(labelName).setPredictionCol(predictionLabel)

    val brierScore = evaluator.getDefaultMetric(evaluator.evaluateAll(dataset_test_df))
    BigDecimal(brierScore).setScale(3, BigDecimal.RoundingMode.HALF_UP).toDouble shouldBe 0.098
  }
}
