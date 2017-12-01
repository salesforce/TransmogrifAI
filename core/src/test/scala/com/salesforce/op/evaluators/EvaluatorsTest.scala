/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.evaluators

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.OpLogisticRegression
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class EvaluatorsTest extends FlatSpec with TestSparkContext {

  val (ds, rawLabel, features) = TestFeatureBuilder[RealNN, OPVector](
    Seq(
      (1.0, Vectors.dense(12.0, 4.3, 1.3)),
      (0.0, Vectors.dense(0.0, 0.3, 0.1)),
      (0.0, Vectors.dense(1.0, 3.9, 4.3)),
      (1.0, Vectors.dense(10.0, 1.3, 0.9)),
      (1.0, Vectors.dense(15.0, 4.7, 1.3)),
      (0.0, Vectors.dense(0.5, 0.9, 10.1)),
      (1.0, Vectors.dense(11.5, 2.3, 1.3)),
      (0.0, Vectors.dense(0.1, 3.3, 0.1))
    ).map(v => v._1.toRealNN -> v._2.toOPVector)
  )
  val label = rawLabel.copy(isResponse = true)


  val (test_ds, test_rawLabel, test_features) = TestFeatureBuilder[RealNN, OPVector](
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
    ).map(v => v._1.toRealNN -> v._2.toOPVector)
  )
  val test_label = test_rawLabel.copy(isResponse = true)

  val testEstimator = new OpLogisticRegression().setInput(label, features)
  val (pred, rawPred, prob) = testEstimator.getOutput()
  val model = testEstimator.fit(ds)
  val transformedData = model.setInput(test_label, test_features).transform(test_ds)


  val sparkBinaryEvaluator = new BinaryClassificationEvaluator().setLabelCol(test_label.name)
    .setRawPredictionCol(rawPred.name)

  val opBinaryMetrics = new OpBinaryClassificationEvaluator().setLabelCol(test_label)
    .setPredictionCol(pred).setRawPredictionCol(rawPred).setProbabilityCol(prob).evaluateAll(transformedData)

  val sparkMultiEvaluator = new MulticlassClassificationEvaluator().setLabelCol(test_label.name)
    .setPredictionCol(pred.name)

  val sparkRegressionEvaluator = new RegressionEvaluator().setLabelCol(test_label.name)
    .setPredictionCol(pred.name)


  "Evaluators" should "have a binary classification factory" in {
    evaluateBinaryMetric(Evaluators.BinaryClassification.auROC()) shouldBe
      evaluateSparkBinaryMetric(BinaryClassEvalMetrics.AuROC.sparkEntryName)

    evaluateBinaryMetric(Evaluators.BinaryClassification.auPR()) shouldBe
      evaluateSparkBinaryMetric(BinaryClassEvalMetrics.AuPR.sparkEntryName)

    evaluateBinaryMetric(Evaluators.BinaryClassification.precision()) shouldBe opBinaryMetrics.Precision
    evaluateBinaryMetric(Evaluators.BinaryClassification.recall()) shouldBe opBinaryMetrics.Recall
    evaluateBinaryMetric(Evaluators.BinaryClassification.f1()) shouldBe opBinaryMetrics.F1
    evaluateBinaryMetric(Evaluators.BinaryClassification.error()) shouldBe opBinaryMetrics.Error
  }

  it should "have a multi classification factory" in {
    evaluateMultiMetric(Evaluators.MultiClassification.precision()) shouldBe
      evaluateSparkMultiMetric(MultiClassEvalMetrics.Precision.sparkEntryName)

    evaluateMultiMetric(Evaluators.MultiClassification.recall()) shouldBe
      evaluateSparkMultiMetric(MultiClassEvalMetrics.Recall.sparkEntryName)

    evaluateMultiMetric(Evaluators.MultiClassification.f1()) shouldBe
      evaluateSparkMultiMetric(MultiClassEvalMetrics.F1.sparkEntryName)

    evaluateMultiMetric(Evaluators.MultiClassification.error()) shouldBe
      1.0 - evaluateSparkMultiMetric(MultiClassEvalMetrics.Error.sparkEntryName)
  }

  it should "have a regression factory" in {
    evaluateRegMetric(Evaluators.Regression.mae()) shouldBe
      evaluateSparkRegMetric(RegressionEvalMetrics.MeanAbsoluteError.sparkEntryName)

    evaluateRegMetric(Evaluators.Regression.mse()) shouldBe
      evaluateSparkRegMetric(RegressionEvalMetrics.MeanSquaredError.sparkEntryName)

    evaluateRegMetric(Evaluators.Regression.rmse()) shouldBe
      evaluateSparkRegMetric(RegressionEvalMetrics.RootMeanSquaredError.sparkEntryName)

    evaluateRegMetric(Evaluators.Regression.r2()) shouldBe
      evaluateSparkRegMetric(RegressionEvalMetrics.R2.sparkEntryName)
  }

  def evaluateBinaryMetric(binEval: OpBinaryClassificationEvaluator): Double = binEval.setLabelCol(test_label)
    .setPredictionCol(pred).setRawPredictionCol(rawPred).setProbabilityCol(prob)
    .evaluate(transformedData)

  def evaluateSparkBinaryMetric(metricName: String): Double = sparkBinaryEvaluator.setMetricName(metricName)
    .evaluate(transformedData)

  def evaluateMultiMetric(multiEval: OpMultiClassificationEvaluator): Double = multiEval.setLabelCol(test_label)
    .setPredictionCol(pred).setRawPredictionCol(rawPred).setProbabilityCol(prob)
    .evaluate(transformedData)

  def evaluateSparkMultiMetric(metricName: String): Double = sparkMultiEvaluator.setMetricName(metricName)
    .evaluate(transformedData)

  def evaluateRegMetric(regEval: OpRegressionEvaluator): Double = regEval.setLabelCol(test_label).setPredictionCol(pred)
    .evaluate(transformedData)

  def evaluateSparkRegMetric(metricName: String): Double = sparkRegressionEvaluator.setMetricName(metricName)
    .evaluate(transformedData)
}

