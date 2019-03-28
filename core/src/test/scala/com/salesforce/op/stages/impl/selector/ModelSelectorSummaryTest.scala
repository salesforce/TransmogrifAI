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

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.evaluators._
import com.salesforce.op.stages.impl.tuning.DataBalancerSummary
import com.salesforce.op.test.TestSparkContext
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ModelSelectorSummaryTest extends FlatSpec with TestSparkContext {

  Spec[ModelSelectorSummary] should "be correctly converted to and from metadata" in {
    val summary = ModelSelectorSummary(
      validationType = ValidationType.CrossValidation,
      validationParameters = Map("testA" -> 5, "otherB" -> Array(1, 2)),
      dataPrepParameters = Map("testB" -> "5", "otherB" -> Seq("1", "2")),
      dataPrepResults = Option(DataBalancerSummary(100L, 300L, 0.1, 2.0, 0.5)),
      evaluationMetric = BinaryClassEvalMetrics.AuROC,
      problemType = ProblemType.BinaryClassification,
      bestModelUID = "test1",
      bestModelName = "test2",
      bestModelType = "test3",
      validationResults = Seq(ModelEvaluation("test4", "test5", "test6", SingleMetric("test7", 0.1), Map.empty)),
      trainEvaluation = BinaryClassificationMetrics(Precision = 0.1, Recall = 0.2, F1 = 0.3, AuROC = 0.4,
        AuPR = 0.5, Error = 0.6, TP = 0.7, TN = 0.8, FP = 0.9, FN = 1.0, thresholds = Seq(1.1),
        precisionByThreshold = Seq(1.2), recallByThreshold = Seq(1.3), falsePositiveRateByThreshold = Seq(1.4),
        BinaryClassificationBinMetrics = BinaryClassificationBinMetrics(
          1.0,
          0.1,
          Seq(0.2, 0.3),
          Seq(3L, 0L, 5L),
          Seq(6.0, 7.0),
          Seq(0.4, 0.5),
          Seq(0.8, 0.9)
        )),
      holdoutEvaluation = Option(RegressionMetrics(RootMeanSquaredError = 1.3, MeanSquaredError = 1.4, R2 = 1.5,
        MeanAbsoluteError = 1.6))
    )

    val meta = summary.toMetadata()
    val decoded = ModelSelectorSummary.fromMetadata(meta)
    decoded.validationType shouldEqual summary.validationType
    decoded.validationParameters.keySet shouldEqual summary.validationParameters.keySet
    decoded.dataPrepParameters.keySet should contain theSameElementsAs summary.dataPrepParameters.keySet
    decoded.dataPrepResults shouldEqual summary.dataPrepResults
    decoded.evaluationMetric.entryName shouldEqual summary.evaluationMetric.entryName
    decoded.problemType shouldEqual summary.problemType
    decoded.bestModelUID shouldEqual summary.bestModelUID
    decoded.bestModelName shouldEqual summary.bestModelName
    decoded.bestModelType shouldEqual summary.bestModelType
    decoded.validationResults shouldEqual summary.validationResults
    decoded.trainEvaluation shouldEqual summary.trainEvaluation
    decoded.holdoutEvaluation shouldEqual summary.holdoutEvaluation
  }

  it should "be correctly converted to and from metadata even when fields are missing or empty" in {

    val summary = ModelSelectorSummary(
      validationType = ValidationType.TrainValidationSplit,
      validationParameters = Map.empty,
      dataPrepParameters = Map.empty,
      dataPrepResults = None,
      evaluationMetric = MultiClassEvalMetrics.Error,
      problemType = ProblemType.Regression,
      bestModelUID = "test1",
      bestModelName = "test2",
      bestModelType = "test3",
      validationResults = Seq.empty,
      trainEvaluation = MultiClassificationMetrics(Precision = 0.1, Recall = 0.2, F1 = 0.3, Error = 0.4,
        ThresholdMetrics = ThresholdMetrics(topNs = Seq(1, 2), thresholds = Seq(1.1, 1.2),
          correctCounts = Map(1 -> Seq(100L)), incorrectCounts = Map(2 -> Seq(200L)),
          noPredictionCounts = Map(3 -> Seq(300L)))),
      holdoutEvaluation = None
    )

    val meta = summary.toMetadata()
    val decoded = ModelSelectorSummary.fromMetadata(meta)
    decoded.validationType shouldEqual summary.validationType
    decoded.validationParameters.keySet shouldEqual summary.validationParameters.keySet
    decoded.dataPrepParameters.keySet should contain theSameElementsAs summary.dataPrepParameters.keySet
    decoded.dataPrepResults shouldEqual summary.dataPrepResults
    decoded.evaluationMetric.entryName shouldEqual summary.evaluationMetric.entryName
    decoded.problemType shouldEqual summary.problemType
    decoded.bestModelUID shouldEqual summary.bestModelUID
    decoded.bestModelName shouldEqual summary.bestModelName
    decoded.bestModelType shouldEqual summary.bestModelType
    decoded.validationResults shouldEqual summary.validationResults
    decoded.trainEvaluation shouldEqual summary.trainEvaluation
    decoded.holdoutEvaluation shouldEqual summary.holdoutEvaluation
  }

}
