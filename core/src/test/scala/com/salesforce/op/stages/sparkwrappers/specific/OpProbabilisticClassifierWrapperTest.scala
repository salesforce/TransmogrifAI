/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.features.types._
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class OpProbabilisticClassifierWrapperTest extends FlatSpec with TestSparkContext {

  val (testData, targetLabel, featureVector) = TestFeatureBuilder("label", "features",
    Seq[(RealNN, OPVector)](
      1.0.toRealNN -> Vectors.dense(12.0, 4.3, 1.3).toOPVector,
      0.0.toRealNN -> Vectors.dense(0.0, 0.3, 0.1).toOPVector,
      0.0.toRealNN -> Vectors.dense(1.0, 3.9, 4.3).toOPVector,
      1.0.toRealNN -> Vectors.dense(10.0, 1.3, 0.9).toOPVector,
      1.0.toRealNN -> Vectors.dense(15.0, 4.7, 1.3).toOPVector,
      0.0.toRealNN -> Vectors.dense(0.5, 0.9, 10.1).toOPVector,
      1.0.toRealNN -> Vectors.dense(11.5, 2.3, 1.3).toOPVector,
      0.0.toRealNN -> Vectors.dense(0.1, 3.3, 0.1).toOPVector,
      0.0.toRealNN -> Vectors.dense(12.0, 3.3, -0.1).toOPVector
    )
  )

  Spec[OpProbabilisticClassifierWrapper[_, _]] should "have the correct params set (fitIntercept = true)" in {
    val lrClassifierModel: LogisticRegressionModel = fitLrModel(fitInterceptParam = true)
    lrClassifierModel.intercept.abs should be > 1E-6
  }

  it should "have the correct params set (logreg with fitIntercept = false)" in {
    val lrClassifierModel: LogisticRegressionModel = fitLrModel(fitInterceptParam = false)
    lrClassifierModel.intercept.abs should be < Double.MinPositiveValue
  }

  it should "should have the expected feature name (decision tree)" in {
    val wrappedEstimator =
      new OpProbabilisticClassifierWrapper[DecisionTreeClassifier, DecisionTreeClassificationModel](
        new DecisionTreeClassifier()
      ).setInput(targetLabel, featureVector)

    val (out1, out2, out3) = wrappedEstimator.getOutput()

    out1.name shouldBe wrappedEstimator.stage1.getOutput().name
    out2.name shouldBe wrappedEstimator.stage2.getOutput().name
    out3.name shouldBe wrappedEstimator.stage3.getOutput().name
  }

  it should "have the correct params set (decision tree with maxDepth = 1)" in {
    val depth = 1
    val dtClassifierModel: DecisionTreeClassificationModel = fitDtModel(depth)
    assert(dtClassifierModel.toDebugString.contains(s"depth $depth"))
  }

  it should "have the correct params set (decision tree with maxDepth = 2)" in {
    val depth = 2
    val dtClassifierModel: DecisionTreeClassificationModel = fitDtModel(depth)
    assert(dtClassifierModel.toDebugString.contains(s"depth $depth"))
  }

  it should "ignore values set for input and output cols outside the OP wrapper" in {
    // configure input classifier and set input col names outside of OP wrapper
    val customLabelColName = "indexedLabel"
    val customFeaturesColName = "indexedFeatures"
    val customProbCol = "xxx"
    val customPredCol = "yyy"
    val customRawCol = "zzz"
    val dtClassifier = new DecisionTreeClassifier()

    dtClassifier.setLabelCol(customLabelColName).setFeaturesCol(customFeaturesColName)
    dtClassifier.setPredictionCol(customPredCol).setProbabilityCol(customProbCol).setRawPredictionCol(customRawCol)

    val dtEstimator =
      new OpProbabilisticClassifierWrapper[DecisionTreeClassifier, DecisionTreeClassificationModel](dtClassifier)
        .setInput(targetLabel, featureVector)

    // verify that the colnames configured outside the opwrapper where ignored and are what is expected
    val inputNames = dtEstimator.stage1.getInputFeatures().map(_.name)
    inputNames should have length 2
    inputNames shouldBe Array(targetLabel.name, featureVector.name)
    dtClassifier.setLabelCol(customLabelColName).setFeaturesCol(customFeaturesColName)
    dtClassifier.setPredictionCol(customPredCol).setProbabilityCol(customProbCol).setRawPredictionCol(customRawCol)

    val model = dtEstimator.fit(testData)

    dtEstimator.uid shouldBe model.uid

    dtClassifier.setLabelCol(customLabelColName).setFeaturesCol(customFeaturesColName)
    dtClassifier.setPredictionCol(customPredCol).setProbabilityCol(customProbCol).setRawPredictionCol(customRawCol)

    val (out1, out2, out3) = model.getOutput()
    val output = model.transform(testData)

    output.schema shouldBe StructType(Array(
      StructField(targetLabel.name, DoubleType, true),
      StructField(featureVector.name, FeatureSparkTypes.sparkTypeOf[OPVector], true),
      StructField(out2.name, FeatureSparkTypes.sparkTypeOf[OPVector], true),
      StructField(out3.name, FeatureSparkTypes.sparkTypeOf[OPVector], true),
      StructField(out1.name, DoubleType, true)
    ))
  }

  def fitDtModel(depth: Int): DecisionTreeClassificationModel = {
    val dtClassifier = new DecisionTreeClassifier().setMaxDepth(depth)

    val dtEstimator = new OpProbabilisticClassifierWrapper[DecisionTreeClassifier, DecisionTreeClassificationModel](
      dtClassifier
    ).setInput(targetLabel, featureVector)

    val model = dtEstimator.fit(testData)
    val output = model.transform(testData)

    val dtClassifierModel = model.stage1.getSparkMlStage().get
    dtClassifierModel
  }

  def fitLrModel(fitInterceptParam: Boolean): LogisticRegressionModel = {
    val regParam = 0.3
    val elasticNetParam = 0.8
    val maxIterParam = 100
    val tolParam = 1E-6

    val lrClassifier = new LogisticRegression()
      .setRegParam(regParam)
      .setElasticNetParam(elasticNetParam)
      .setMaxIter(maxIterParam)
      .setTol(tolParam)
      .setFitIntercept(fitInterceptParam)

    val testEstimator = new OpProbabilisticClassifierWrapper[LogisticRegression, LogisticRegressionModel](
      lrClassifier
    ).setInput(targetLabel, featureVector)

    val model = testEstimator.fit(testData)
    val output = model.transform(testData)

    val lrClassifierModel = model.stage1.getSparkMlStage().get

    lrClassifierModel.getRegParam shouldBe regParam
    lrClassifierModel.getElasticNetParam shouldBe elasticNetParam
    lrClassifierModel.getMaxIter shouldBe maxIterParam
    lrClassifierModel.getTol shouldBe tolParam
    lrClassifierModel.getFitIntercept shouldBe fitInterceptParam

    lrClassifierModel
  }
}


