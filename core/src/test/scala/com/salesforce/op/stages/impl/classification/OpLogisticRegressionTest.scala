/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.features.types._
import com.salesforce.op.stages.sparkwrappers.generic._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class OpLogisticRegressionTest extends FlatSpec with TestSparkContext {

  val (testData, rawFeature1, feature2) = TestFeatureBuilder("label", "features",
    Seq[(RealNN, OPVector)](
      1.0.toRealNN -> Vectors.dense(12.0, 4.3, 1.3).toOPVector,
      0.0.toRealNN -> Vectors.dense(0.0, 0.3, 0.1).toOPVector,
      0.0.toRealNN -> Vectors.dense(1.0, 3.9, 4.3).toOPVector,
      1.0.toRealNN -> Vectors.dense(10.0, 1.3, 0.9).toOPVector,
      1.0.toRealNN -> Vectors.dense(15.0, 4.7, 1.3).toOPVector,
      0.0.toRealNN -> Vectors.dense(0.5, 0.9, 10.1).toOPVector,
      1.0.toRealNN -> Vectors.dense(11.5, 2.3, 1.3).toOPVector,
      0.0.toRealNN -> Vectors.dense(0.1, 3.3, 0.1).toOPVector
    )
  )
  val feature1 = rawFeature1.copy(isResponse = true)
  val logReg = new OpLogisticRegression().setInput(feature1, feature2)

  Spec[OpLogisticRegression] should "have properly formed stage1" in {
    assert(logReg.stage1.isInstanceOf[SwBinaryEstimator[_, _, _, _, _]])
    val inputNames = logReg.stage1.getInputFeatures().map(_.name)
    inputNames should have length 2
    inputNames shouldBe Array(feature1.name, feature2.name)
    logReg.stage1.getOutput().name shouldBe logReg.stage1.getOutputFeatureName
    the[IllegalArgumentException] thrownBy {
      logReg.setInput(feature1.copy(isResponse = true), feature2.copy(isResponse = true))
    } should have message "The feature vector should not contain any response features."
  }

  it should "have properly formed stage2" in {
    assert(logReg.stage2.isInstanceOf[SwTernaryTransformer[_, _, _, _, _]])
    val inputNames = logReg.stage2.getInputFeatures().map(_.name)
    inputNames should have length 3
    inputNames shouldBe Array(feature1.name, feature2.name, logReg.stage1.getOutputFeatureName)
    logReg.stage2.getOutput().name shouldBe logReg.stage2.getOutputFeatureName

  }

  it should "have properly formed stage3" in {
    assert(logReg.stage3.isInstanceOf[SwQuaternaryTransformer[_, _, _, _, _, _]])
    val inputNames = logReg.stage3.getInputFeatures().map(_.name)
    inputNames should have length 4
    inputNames shouldBe Array(feature1.name, feature2.name, logReg.stage1.getOutputFeatureName,
      logReg.stage2.getOutputFeatureName)

    logReg.stage3.getOutput().name shouldBe logReg.stage3.getOutputFeatureName
  }

  it should "have proper outputs corresponding to the stages" in {
    val outputs = logReg.getOutput()
    outputs._1.name shouldBe logReg.stage1.getOutput().name
    outputs._2.name shouldBe logReg.stage2.getOutput().name
    outputs._3.name shouldBe logReg.stage3.getOutput().name

    // as long as the parent stages are correct, we can also assume
    // that the parent features are correct, since that should
    // be verified in the unit tests for the transformers.
    outputs._1.originStage shouldBe logReg.stage1
    outputs._2.originStage shouldBe logReg.stage2
    outputs._3.originStage shouldBe logReg.stage3
  }


  it should "return a properly formed LogisticRegressionModel when fitted" in {
    val model = logReg.setSparkParams("maxIter", 10).fit(testData)

    model shouldBe a[SwThreeStageBinaryModel[_, _, _, _, _, _]]
    model.stage1 shouldBe a[SwBinaryModel[_, _, _, _]]

    val sparkStage = model.stage1.getSparkMlStage()
    sparkStage.get.isInstanceOf[LogisticRegressionModel]
    assert(model.stage2.getSparkMlStage().isEmpty)
    assert(model.stage3.getSparkMlStage().isEmpty)

    model.stage1OperationName shouldBe "LogisticRegression_predictionCol"
    model.stage2OperationName shouldBe "LogisticRegression_rawPredictionCol"
    model.stage3OperationName shouldBe "LogisticRegression_probabilityCol"

    val inputNames = model.getInputFeatures().map(_.name)
    inputNames should have length 2
    inputNames shouldBe Array(feature1.name, feature2.name)
  }


  it should "allow the user to set the desired spark parameters" in {
    logReg.setSparkParams("maxIter", 10).setSparkParams("regParam", 0.1)
    logReg.getSparkParams("maxIter") shouldBe Some(10)
    logReg.getSparkParams("regParam") shouldBe Some(0.1)

    logReg.setThresholds(Array(0.03, 0.06)).setElasticNetParam(0.1)
    logReg.getSparkParams("thresholds").get.asInstanceOf[Array[Double]] should contain theSameElementsAs
      Array(0.03, 0.06)
    logReg.getSparkParams("elasticNetParam") shouldBe Some(0.1)
  }

  // TODO: move this to OpWorkFlowTest
  //  it should "work in a workflow" in {
  //    val (prob, rawpred, pred) = logReg.getOutput()
  //    val workflow = new OpWorkflow().setResultFeatures(pred)
  //
  //    val reader = DataReaders.Simple.custom[LRDataTest](
  //      readFn = (s: Option[String], spk: SparkSession) => spk.sparkContext.parallelize(DataTest.input)
  //    )
  //
  //    val workflowModel = workflow.setReader(reader).train()
  //    val scores = workflowModel.score()
  //    val justScores = scores.select(s"(label)_(features)_((label)_(features)_${stageNames(0)})_" +
  //      s"((label)_(features)_((label)_(features)_${stageNames(0)})_${stageNames(1)})_${stageNames(2)}")
  //      .collect().map(_.getAs[Double](0)).toList
  //    justScores shouldEqual List(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0)
  //  }

}


