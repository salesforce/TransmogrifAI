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

import com.salesforce.op.evaluators.BinaryClassEvalMetrics._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.{BinaryClassificationModelSelector, OpLogisticRegression}
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.EstimatorType
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


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

  val lr = new OpLogisticRegression()
  val lrParams = new ParamGridBuilder().addGrid(lr.regParam, Array(0.0)).build()

  val testEstimator = BinaryClassificationModelSelector.withTrainValidationSplit(splitter = None, trainRatio = 0.5,
    modelsAndParameters = Seq(lr -> lrParams))
    .setInput(label, features)
  val pred = testEstimator.getOutput()
  val model = testEstimator.fit(ds)

  val testEvaluator = new OpBinaryClassificationEvaluator().setLabelCol(label)
    .setPredictionCol(pred)

  // with single predicition putput
  val testEstimator2 = new OpLogisticRegression().setInput(label, features)
  val prediction = testEstimator2.getOutput()
  val model2 = testEstimator2.fit(ds)

  val testEvaluator2 = new OpBinaryClassificationEvaluator().setLabelCol(label)
    .setPredictionCol(prediction)

  // comparisons
  val sparkBinaryEvaluator = new BinaryClassificationEvaluator()
  val sparkMulticlassEvaluator = new MulticlassClassificationEvaluator()

  val rawPred = pred.map[OPVector](p => Vectors.dense(p.rawPrediction).toOPVector)
  val predValue = pred.map[RealNN](_.prediction.toRealNN)

  val transformedData = model.setInput(test_label, test_features).transform(test_ds)
  val flattenedData1 = rawPred.originStage.asInstanceOf[Transformer].transform(transformedData)
  val flattenedData2 = predValue.originStage.asInstanceOf[Transformer].transform(flattenedData1)

  sparkBinaryEvaluator.setLabelCol(label.name).setRawPredictionCol(rawPred.name)
  sparkMulticlassEvaluator.setLabelCol(label.name).setPredictionCol(predValue.name)

  Spec[OpBinaryClassificationEvaluator] should "copy" in {
    val testEvaluatorCopy = testEvaluator.copy(ParamMap())
    testEvaluatorCopy.uid shouldBe testEvaluator.uid
  }


  it should "evaluate the metrics with three inputs" in {
    val metrics = testEvaluator.evaluateAll(transformedData)

    val (tp, tn, fp, fn, precision, recall, f1) = getPosNegValues(
      flattenedData2.select(predValue.name, test_label.name).rdd
    )

    tp.toDouble shouldBe metrics.TP
    tn.toDouble shouldBe metrics.TN
    fp.toDouble shouldBe metrics.FP
    fn.toDouble shouldBe metrics.FN

    precision shouldBe metrics.Precision
    recall shouldBe metrics.Recall
    f1 shouldBe metrics.F1
    1.0 - sparkMulticlassEvaluator.setMetricName(Error.sparkEntryName).evaluate(flattenedData2) shouldBe metrics.Error
  }

  it should "evaluate the metrics with one prediction input" in {
    val transformedData2 = model2.setInput(test_label, test_features).transform(test_ds)
    val metrics = testEvaluator2.evaluateAll(transformedData2)

    val (tp, tn, fp, fn, precision, recall, f1) = getPosNegValues(
      transformedData2.select(prediction.name, test_label.name).rdd
        .map( r => Row(r.getMap[String, Double](0).toMap.toPrediction.prediction, r.getDouble(1)) )
    )

    tp.toDouble shouldBe metrics.TP
    tn.toDouble shouldBe metrics.TN
    fp.toDouble shouldBe metrics.FP
    fn.toDouble shouldBe metrics.FN

    metrics.Precision shouldBe precision
    metrics.Recall shouldBe recall
    metrics.F1 shouldBe f1
  }

  it should "evaluate the metrics on dataset with only the label and prediction 0" in {
    model.setInput(zero_label, zero_features)
    val transformedDataZero = model.transform(zero_ds)
    val outPred = model.getOutput()
    val metricsZero = testEvaluator.setLabelCol(zero_rawLabel).setPredictionCol(outPred)
      .evaluateAll(transformedDataZero)

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
