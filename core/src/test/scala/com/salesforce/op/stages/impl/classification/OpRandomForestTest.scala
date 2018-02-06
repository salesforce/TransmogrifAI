/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.Impurity.Gini
import com.salesforce.op.stages.sparkwrappers.generic._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class OpRandomForestTest extends FlatSpec with TestSparkContext {

  val stageNames = Array[String]("RandomForestClassifier_predictionCol", "RandomForestClassifier_rawPredictionCol",
    "RandomForestClassifier_probabilityCol"
  )

  lazy val (testData, rawLabel, features) = TestFeatureBuilder[RealNN, OPVector]("label", "features",
    Seq(
      (1.0.toRealNN, Vectors.dense(12.0, 4.3, 1.3).toOPVector),
      (0.0.toRealNN, Vectors.dense(0.0, 0.3, 0.1).toOPVector),
      (0.0.toRealNN, Vectors.dense(1.0, 3.9, 4.3).toOPVector),
      (1.0.toRealNN, Vectors.dense(10.0, 1.3, 0.9).toOPVector),
      (1.0.toRealNN, Vectors.dense(15.0, 4.7, 1.3).toOPVector),
      (0.0.toRealNN, Vectors.dense(0.5, 0.9, 10.1).toOPVector),
      (1.0.toRealNN, Vectors.dense(11.5, 2.3, 1.3).toOPVector),
      (0.0.toRealNN, Vectors.dense(0.1, 3.3, 0.1).toOPVector)
    )
  )

  val label = rawLabel.copy(isResponse = true)

  lazy val (multiClassTestData, rawLabelMulti, featuresMulti) =
    TestFeatureBuilder[RealNN, OPVector]("labelMulti", "featuresMulti",
      Seq(
        (1.0.toRealNN, Vectors.dense(12.0, 4.3, 1.3).toOPVector),
        (0.0.toRealNN, Vectors.dense(0.0, 0.3, 0.1).toOPVector),
        (2.0.toRealNN, Vectors.dense(1.0, 3.9, 4.3).toOPVector),
        (2.0.toRealNN, Vectors.dense(10.0, 1.3, 0.9).toOPVector),
        (1.0.toRealNN, Vectors.dense(15.0, 4.7, 1.3).toOPVector),
        (0.0.toRealNN, Vectors.dense(0.5, 0.9, 10.1).toOPVector),
        (1.0.toRealNN, Vectors.dense(11.5, 2.3, 1.3).toOPVector),
        (0.0.toRealNN, Vectors.dense(0.1, 3.3, 0.1).toOPVector),
        (2.0.toRealNN, Vectors.dense(1.0, 4.0, 4.5).toOPVector),
        (2.0.toRealNN, Vectors.dense(10.0, 1.5, 1.0).toOPVector)
      )
    )

  val labelMulti = rawLabelMulti.copy(isResponse = true)

  val randomForest = new OpRandomForest().setInput(label, features)
  val outputs = randomForest.getOutput()
  val (predName, rawName, probName) = (outputs._1.name, outputs._2.name, outputs._3.name)

  val randomForestMulti = new OpRandomForest().setInput(labelMulti, featuresMulti)
  val outputsMulti = randomForestMulti.getOutput()
  val (predNameMulti, rawNameMulti, probNameMulti) = (outputsMulti._1.name, outputsMulti._2.name, outputsMulti._3.name)

  Spec[OpRandomForest] should "allow the user to set the desired spark parameters" in {
    randomForest.setThresholds(Array(1.0, 1.0))
      .setMaxDepth(10)
      .setImpurity(Impurity.Gini)
      .setMaxBins(33)
      .setMinInstancesPerNode(2)
      .setMinInfoGain(0.2)
      .setSubsamplingRate(0.9)
      .setNumTrees(21)
      .setSeed(2L)

    randomForest.getSparkParams("thresholds").get.asInstanceOf[Array[Double]] should
      contain theSameElementsAs Array(1.0, 1.0)
    randomForest.getSparkParams("maxDepth").get.asInstanceOf[Int] shouldBe 10
    randomForest.getSparkParams("maxBins").get.asInstanceOf[Int] shouldBe 33
    randomForest.getSparkParams("impurity").get.asInstanceOf[String] shouldBe Impurity.Gini.sparkName
    randomForest.getSparkParams("minInstancesPerNode").get.asInstanceOf[Int] shouldBe 2
    randomForest.getSparkParams("minInfoGain").get.asInstanceOf[Double] shouldBe 0.2
    randomForest.getSparkParams("subsamplingRate").get.asInstanceOf[Double] shouldBe 0.9
    randomForest.getSparkParams("numTrees").get.asInstanceOf[Int] shouldBe 21
    randomForest.getSparkParams("seed").get.asInstanceOf[Long] shouldBe 2L
  }

  it should "return a properly formed Random Forest when fitted" in {
    the[IllegalArgumentException] thrownBy {
      randomForest.setInput(label.copy(isResponse = true), features.copy(isResponse = true))
    } should have message "The feature vector should not contain any response features."

    val model = randomForest.fit(testData)

    model shouldBe a[SwThreeStageBinaryModel[_, _, _, _, _, _]]
    model.stage1 shouldBe a[SwBinaryModel[_, _, _, _]]

    val sparkStage = model.stage1.getSparkMlStage()
    assert(sparkStage.get.isInstanceOf[RandomForestClassificationModel])
    assert(model.stage2.getSparkMlStage().isEmpty)
    assert(model.stage3.getSparkMlStage().isEmpty)

    model.stage1OperationName shouldBe stageNames(0)
    model.stage2OperationName shouldBe stageNames(1)
    model.stage3OperationName shouldBe stageNames(2)

    val inputNames = model.getInputFeatures().map(_.name)
    inputNames should have length 2
    inputNames shouldBe Array(label.name, features.name)

    val transformedData = model.transform(testData)

    val fields = transformedData.select(rawName, probName, predName).schema.fields

    fields.map(_.name).toList shouldBe List(rawName, probName, predName)

    fields.map(_.dataType.typeName).toList shouldBe List("vector", "vector", "double")
  }

  it should "be implemented using shortcuts" in {
    val (raw, prob, pred) = features.randomForest(label = label, impurity = Gini)
    raw.name shouldBe raw.originStage.outputName
    prob.name shouldBe prob.originStage.outputName
    pred.name shouldBe pred.originStage.outputName
  }

  it should "return a model for multiClassification problem" in {
    the[IllegalArgumentException] thrownBy {
      randomForestMulti.setInput(labelMulti.copy(isResponse = true), featuresMulti.copy(isResponse = true))
    } should have message "The feature vector should not contain any response features."

    val modelMulti = randomForestMulti.fit(multiClassTestData)
    val transformedDataMulti = modelMulti.transform(multiClassTestData)
    val fieldsMulti = transformedDataMulti.select(rawNameMulti,
      probNameMulti, predNameMulti).schema.fields
    fieldsMulti.map(_.name).toList shouldBe List(rawNameMulti, probNameMulti, predNameMulti)
    fieldsMulti.map(_.dataType.typeName).toList shouldBe List("vector", "vector", "double")
  }
}
