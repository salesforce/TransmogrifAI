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
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.TableDrivenPropertyChecks._


@RunWith(classOf[JUnitRunner])
class EvaluatorsTest extends FlatSpec with TestSparkContext {

  val (trainData, trainRawLabel, trainFeatures) = TestFeatureBuilder[RealNN, OPVector](
    Seq(
      (1.0, Vectors.dense(12.0, 4.3, 1.3)),
      (0.0, Vectors.dense(0.0, 0.3, 0.1)),
      (0.0, Vectors.dense(1.0, 3.9, 4.3)),
      (1.0, Vectors.dense(10.0, 1.3, 0.9)),
      (1.0, Vectors.dense(15.0, 4.7, 1.3)),
      (0.0, Vectors.dense(0.5, 0.9, 10.1)),
      (1.0, Vectors.dense(11.5, 2.3, 1.3)),
      (0.0, Vectors.dense(0.1, 3.3, 0.1))
    ).map { case (lbl, feats) => lbl.toRealNN -> feats.toOPVector }
  )
  val trainLabel = trainRawLabel.copy(isResponse = true)


  val (testData, testRawLabel, testFeatures) = TestFeatureBuilder[RealNN, OPVector](
    Seq(
      (1.0, Vectors.dense(3.0, 54.4, 46.9)),
      (0.0, Vectors.dense(4.0, 300, 90)),
      (0.0, Vectors.dense(3.0, 4.0, 4.43)),
      (1.0, Vectors.dense(1.0, 41.3, -0.9)),
      (1.0, Vectors.dense(5.0, 43.7, 91.3)),
      (0.0, Vectors.dense(6, -10.9, -10.1)),
      (1.0, Vectors.dense(14.5, 2.35, -1.3)),
      (0.0, Vectors.dense(6.3, 30.3, -0.1)),
      (1.0, Vectors.dense(-15.0, 64.7, -1.3)),
      (0.0, Vectors.dense(-0.5, 0, -1.1)),
      (1.0, Vectors.dense(-11.5, -22.3, -41.3)),
      (0.0, Vectors.dense(-0.1, 63.3, 3.1)),
      (1.0, Vectors.dense(115.0, 54.7, 4.3)),
      (0.0, Vectors.dense(20.5, -0.34, 50.1)),
      (1.0, Vectors.dense(411.5, 2.54, 6.3)),
      (0.0, Vectors.dense(50.1, -3.3, 6.1))
    ).map { case (lbl, feats) => lbl.toRealNN -> feats.toOPVector }
  )
  val testLabel = testRawLabel.copy(isResponse = true)

  val testEstimator = new OpLogisticRegression().setInput(trainLabel, trainFeatures)
  val pred = testEstimator.getOutput()
  val model = testEstimator.fit(trainData)


  val transformedData = model.setInput(testLabel, testFeatures).transform(testData)
  val rawPred = pred.map[OPVector](p => Vectors.dense(p.rawPrediction).toOPVector)
  val predValue = pred.map[RealNN](_.prediction.toRealNN)

  val evaluationData = {
    val transformed = rawPred.originStage.asInstanceOf[Transformer].transform(transformedData)
    predValue.originStage.asInstanceOf[Transformer].transform(transformed)
  }
  val emptyData = evaluationData.filter(r => r.get(0) == "blarg")

  val sparkBinaryEvaluator = new BinaryClassificationEvaluator().setLabelCol(testLabel.name)
    .setRawPredictionCol(rawPred.name)

  val opBinaryMetrics = new OpBinaryClassificationEvaluator().setLabelCol(testLabel)
    .setPredictionCol(pred).evaluateAll(transformedData)

  val opBinScoreMetrics = new OpBinScoreEvaluator().setLabelCol(testLabel)
    .setPredictionCol(pred).evaluateAll(transformedData)

  val sparkMultiEvaluator = new MulticlassClassificationEvaluator().setLabelCol(testLabel.name)
    .setPredictionCol(predValue.name)

  val sparkRegressionEvaluator = new RegressionEvaluator().setLabelCol(testLabel.name)
    .setPredictionCol(predValue.name)


  Spec(Evaluators.getClass) should "have a binary classification factory" in {
    val binaryEvaluators = Table(
      ("evaluator", "expectedValue", "expectedDefault"),
      (Evaluators.BinaryClassification(), evalSparkBinary(BinaryClassEvalMetrics.AuROC.sparkEntryName), 0.0),
      (Evaluators.BinaryClassification.auROC(), evalSparkBinary(BinaryClassEvalMetrics.AuROC.sparkEntryName), 0.0),
      (Evaluators.BinaryClassification.brierScore(), opBinScoreMetrics.brierScore, 0.0),
      (Evaluators.BinaryClassification.auPR(), evalSparkBinary(BinaryClassEvalMetrics.AuPR.sparkEntryName), 0.0),
      (Evaluators.BinaryClassification.precision(), opBinaryMetrics.Precision, 0.0),
      (Evaluators.BinaryClassification.recall(), opBinaryMetrics.Recall, 0.0),
      (Evaluators.BinaryClassification.f1(), opBinaryMetrics.F1, 0.0),
      (Evaluators.BinaryClassification.error(), opBinaryMetrics.Error, 0.0),
      (Evaluators.BinaryClassification.custom(metricName = "test", evaluateFn = data => -123.0), -123.0, -123.0)
    )

    forAll(binaryEvaluators) { case (evaluator, expectedValue, expectedDefault) =>
      evaluator shouldBe a[OpBinaryClassificationEvaluatorBase[_]]
      evaluator.setLabelCol(testLabel).setPredictionCol(pred).evaluate(evaluationData) shouldBe expectedValue
      evaluator.evaluate(emptyData) shouldBe expectedDefault
    }
  }

  it should "have a multi classification factory" in {
    val multiEvaluators = Table(
      ("evaluator", "expectedValue", "expectedDefault"),
      (Evaluators.MultiClassification(), evalSparkMulti(MultiClassEvalMetrics.F1.sparkEntryName), 0.0),
      (Evaluators.MultiClassification.precision(), evalSparkMulti(MultiClassEvalMetrics.Precision.sparkEntryName), 0.0),
      (Evaluators.MultiClassification.recall(), evalSparkMulti(MultiClassEvalMetrics.Recall.sparkEntryName), 0.0),
      (Evaluators.MultiClassification.f1(), evalSparkMulti(MultiClassEvalMetrics.F1.sparkEntryName), 0.0),
      (Evaluators.MultiClassification.error(), 1.0 - evalSparkMulti(MultiClassEvalMetrics.Error.sparkEntryName), 0.0),
      (Evaluators.MultiClassification.custom(metricName = "test", evaluateFn = data => -456.0), -456.0, -456.0)
    )

    forAll(multiEvaluators) { case (evaluator, expectedValue, expectedDefault) =>
      evaluator shouldBe a[OpMultiClassificationEvaluatorBase[_]]
      evaluator.setLabelCol(testLabel).setPredictionCol(pred).evaluate(evaluationData) shouldBe expectedValue
      evaluator.evaluate(emptyData) shouldBe expectedDefault
    }
  }

  it should "have a regression factory" in {
    val regrEvaluators = Table(
      ("evaluator", "expectedValue", "expectedDefault"),
      (Evaluators.Regression(), evalSparkRegr(RegressionEvalMetrics.RootMeanSquaredError.sparkEntryName), 0.0),
      (Evaluators.Regression.mae(), evalSparkRegr(RegressionEvalMetrics.MeanAbsoluteError.sparkEntryName), 0.0),
      (Evaluators.Regression.mse(), evalSparkRegr(RegressionEvalMetrics.MeanSquaredError.sparkEntryName), 0.0),
      (Evaluators.Regression.rmse(), evalSparkRegr(RegressionEvalMetrics.RootMeanSquaredError.sparkEntryName), 0.0),
      (Evaluators.Regression.r2(), evalSparkRegr(RegressionEvalMetrics.R2.sparkEntryName), 0.0),
      (Evaluators.Regression.custom(metricName = "test", evaluateFn = data => -789.0), -789.0, -789.0)
    )

    forAll(regrEvaluators) { case (evaluator, expectedValue, expectedDefault) =>
      evaluator shouldBe a[OpRegressionEvaluatorBase[_]]
      evaluator.setLabelCol(testLabel).setPredictionCol(pred).evaluate(evaluationData) shouldBe expectedValue
      evaluator.evaluate(emptyData) shouldBe expectedDefault
    }
  }

  def evalSparkBinary(metricName: String): Double =
    sparkBinaryEvaluator.setMetricName(metricName).evaluate(evaluationData)

  def evalSparkMulti(metricName: String): Double =
    sparkMultiEvaluator.setMetricName(metricName).evaluate(evaluationData)

  def evalSparkRegr(metricName: String): Double =
    sparkRegressionEvaluator.setMetricName(metricName).evaluate(evaluationData)
}

