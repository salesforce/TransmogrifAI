/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.preparators.SanityChecker
import com.salesforce.op.stages.sparkwrappers.generic._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegressionModel
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class OpLinearRegressionTest extends FlatSpec with TestSparkContext {
  val stageNames = Array[String]("LinearRegression_predictionCol")

  val (ds, rawLabel, features) = TestFeatureBuilder(
    Seq[(RealNN, OPVector)](
      (10.0.toRealNN, Vectors.dense(1.0, 4.3, 1.3).toOPVector),
      (20.0.toRealNN, Vectors.dense(2.0, 0.3, 0.1).toOPVector),
      (30.0.toRealNN, Vectors.dense(3.0, 3.9, 4.3).toOPVector),
      (40.0.toRealNN, Vectors.dense(4.0, 1.3, 0.9).toOPVector),
      (50.0.toRealNN, Vectors.dense(5.0, 4.7, 1.3).toOPVector)
      )
  )
  val label = rawLabel.copy(isResponse = true)
  val linReg = new OpLinearRegression().setInput(label, features)

  Spec[OpLinearRegression] should "have output with correct origin stage" in {
    val output = linReg.getOutput()
    assert(output.originStage.isInstanceOf[SwBinaryEstimator[_, _, _, _, _]])
    the[IllegalArgumentException] thrownBy {
      linReg.setInput(label.copy(isResponse = true), features.copy(isResponse = true))
    } should have message "The feature vector should not contain any response features."
  }

  it should "return a properly formed LinearRegressionModel when fitted" in {
    val model = linReg.setSparkParams("maxIter", 10).fit(ds)
    assert(model.isInstanceOf[SwBinaryModel[RealNN, OPVector, RealNN, LinearRegressionModel]])

    val sparkStage = model.getSparkMlStage()

    sparkStage.isDefined shouldBe true
    sparkStage.get shouldBe a[LinearRegressionModel]

    val inputNames = model.getInputFeatures().map(_.name)
    inputNames should have length 2
    inputNames shouldBe Array(label.name, features.name)
  }


  it should "allow the user to set the desired spark parameters" in {
    linReg.setMaxIter(10).setRegParam(0.1)
    linReg.getMaxIter shouldBe 10
    linReg.getRegParam shouldBe 0.1

    linReg.setFitIntercept(true).setElasticNetParam(0.1).setSolver("normal")
    linReg.getFitIntercept shouldBe true
    linReg.getElasticNetParam shouldBe 0.1
    linReg.getSolver shouldBe "normal"
  }
}
