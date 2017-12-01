/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.evaluators

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.regression.OpLinearRegression
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class OpRegressionEvaluatorTest extends FlatSpec with TestSparkContext {

  val (ds, rawLabel, features) = TestFeatureBuilder[RealNN, OPVector](
    Seq(
      (10.0, Vectors.dense(1.0, 4.3, 1.3)),
      (20.0, Vectors.dense(2.0, 0.3, 0.1)),
      (30.0, Vectors.dense(3.0, 3.9, 4.3)),
      (40.0, Vectors.dense(4.0, 1.3, 0.9)),
      (50.0, Vectors.dense(5.0, 4.7, 1.3))
    ).map(v => v._1.toRealNN -> v._2.toOPVector)
  )

  val label = rawLabel.copy(isResponse = true)
  val testEstimator = new OpLinearRegression().setInput(label, features)

  val prediction = testEstimator.getOutput()
  val testEvaluator = new OpRegressionEvaluator().setLabelCol(label).setPredictionCol(prediction)

  Spec[OpRegressionEvaluator] should "copy" in {
    val testEvaluatorCopy = testEvaluator.copy(ParamMap())
    testEvaluatorCopy.uid shouldBe testEvaluator.uid
  }

  it should "evaluate the metrics" in {
    val model = testEstimator.fit(ds)
    val transformedData = model.setInput(label, features).transform(ds)
    val metrics = testEvaluator.evaluateAll(transformedData).toMetadata

    assert(metrics.getDouble("RootMeanSquaredError") <= 1E-12, "rmse should be close to 0")
    assert(metrics.getDouble("MeanSquaredError") <= 1E-24, "mse should be close to 0")
    assert(metrics.getDouble("R2") == 1.0, "R2 should equal 1.0")
    assert(metrics.getDouble("MeanAbsoluteError") <= 1E-12, "mae should be close to 0")
  }
}
