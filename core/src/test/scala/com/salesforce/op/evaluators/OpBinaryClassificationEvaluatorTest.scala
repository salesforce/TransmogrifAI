/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.evaluators

import com.salesforce.op.evaluators.BinaryClassEvalMetrics._
import com.salesforce.op.evaluators.MultiClassEvalMetrics._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.OpLogisticRegression
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class OpBinaryClassificationEvaluatorTest extends FlatSpec with TestSparkContext {

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

  val (zero_ds, zero_rawLabel, zero_features) = TestFeatureBuilder[RealNN, OPVector](
    Seq(
      (0.0, Vectors.dense(4.0, 300, 90))
    ).map(v => v._1.toRealNN -> v._2.toOPVector)
  )
  val zero_label = zero_rawLabel.copy(isResponse = true)


  val (one_ds, one_rawLabel, one_features) = TestFeatureBuilder[RealNN, OPVector](
    Seq(
      (1.0, Vectors.dense(3.0, 54.4, 46.9))
    ).map(v => v._1.toRealNN -> v._2.toOPVector)
  )
  val one_label = one_rawLabel.copy(isResponse = true)

  val testEstimator = new OpLogisticRegression().setInput(label, features)
  val (pred, rawPred, _) = testEstimator.getOutput()
  val testEvaluator = new OpBinaryClassificationEvaluator().setLabelCol(label)
    .setPredictionCol(pred)
    .setRawPredictionCol(rawPred)
  val model = testEstimator.fit(ds)
  val sparkBinaryEvaluator = new BinaryClassificationEvaluator()
  val sparkMulticlassEvaluator = new MulticlassClassificationEvaluator()
  Spec[OpBinaryClassificationEvaluator] should "copy" in {
    val testEvaluatorCopy = testEvaluator.copy(ParamMap())
    testEvaluatorCopy.uid shouldBe testEvaluator.uid
  }

  it should "evaluate the metrics" in {
    val transformedData = model.setInput(test_label, test_features).transform(test_ds)
    val metrics = testEvaluator.evaluateAll(transformedData)

    sparkBinaryEvaluator.setLabelCol(label.name).setRawPredictionCol(rawPred.name)
    sparkMulticlassEvaluator.setLabelCol(label.name).setPredictionCol(pred.name)

    metrics.AuROC shouldBe sparkBinaryEvaluator.setMetricName(AuROC.sparkEntryName).evaluate(transformedData)
    metrics.AuPR shouldBe sparkBinaryEvaluator.setMetricName(AuPR.sparkEntryName).evaluate(transformedData)

    val (tp, tn, fp, fn, precision, recall, f1) = getPosNegValues(
      transformedData.select(pred.name, test_label.name).rdd
    )

    tp.toDouble shouldBe metrics.TP
    tn.toDouble shouldBe metrics.TN
    fp.toDouble shouldBe metrics.FP
    fn.toDouble shouldBe metrics.FN

    metrics.Precision shouldBe precision
    metrics.Recall shouldBe recall
    metrics.F1 shouldBe f1
    metrics.Error shouldBe 1.0 - sparkMulticlassEvaluator.setMetricName(Error.sparkEntryName).evaluate(transformedData)
  }


  it should "evaluate the metrics on dataset with only the label and prediction 0" in {
    val transformedDataZero = model.setInput(zero_label, zero_features).transform(zero_ds)

    val metricsZero = testEvaluator.setLabelCol(zero_rawLabel).evaluateAll(transformedDataZero)

    sparkMulticlassEvaluator.setLabelCol(zero_label.name).setPredictionCol(pred.name)

    metricsZero.TN shouldBe 1.0
    metricsZero.TP shouldBe 0.0
    metricsZero.FN shouldBe 0.0
    metricsZero.FP shouldBe 0.0
    metricsZero.Precision shouldBe 0.0
    metricsZero.Recall shouldBe 0.0
    metricsZero.Error shouldBe 0.0
  }



  it should "evaluate the metrics on dataset with only the label and prediction 1" in {
    val transformedDataOne = model.setInput(one_label, one_features).transform(one_ds)

    val metricsOne = testEvaluator.setLabelCol(one_label).evaluateAll(transformedDataOne)

    sparkMulticlassEvaluator.setLabelCol(one_label.name).setPredictionCol(pred.name)

    metricsOne.TN shouldBe 0.0
    metricsOne.TP shouldBe 1.0
    metricsOne.FN shouldBe 0.0
    metricsOne.FP shouldBe 0.0
    metricsOne.Precision shouldBe 1.0
    metricsOne.Recall shouldBe 1.0
    metricsOne.Error shouldBe 0.0
  }

  private def getPosNegValues(rdd: RDD[Row]): (Double, Double, Double, Double, Double, Double, Double) = {
    val metric = rdd.map(row => (
      if (row.getDouble(0) > 0.0 && row.getDouble(1) > 0.0) 1 else 0, // True Positive
      if (row.getDouble(0) < 1.0 && row.getDouble(1) < 1.0) 1 else 0, // True Negative
      if (row.getDouble(0) > 0.0 && row.getDouble(1) < 1.0) 1 else 0, // False Positive
      if (row.getDouble(0) < 1.0 && row.getDouble(1) > 0.0) 1 else 0)) // False Negative
      .reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4))

    val tp = metric._1.toDouble
    val tn = metric._2.toDouble
    val fp = metric._3.toDouble
    val fn = metric._4.toDouble

    val multiclassMetrics = new MulticlassMetrics(rdd.map(row => (row.getDouble(0), row.getDouble(1))))

    val (precision, recall, f1) = (multiclassMetrics.precision(1.0), multiclassMetrics.recall(1.0),
      multiclassMetrics.fMeasure(1.0))

    (tp, tn, fp, fn, precision, recall, f1)

  }


  // TODO: move this to OpWorkFlowTest
  //  it should "evaluate  a  workflow" in {
  //    val workflow = new OpWorkflow().setResultFeatures(rawPred, pred)
  //    val reader = DataReaders.Simple.custom[LRDataTest](
  //      readFn = (s: Option[String], spk: SparkSession) => spk.sparkContext.parallelize(DataTest.input)
  //    )
  //    val workflowModel = workflow.setReader(reader).train()(spark)
  //    val metricsScore = workflowModel.evaluate(testEvaluator)(spark)
  //
  //    metricsScore shouldBe BinaryClassificationMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
  //  }
}
