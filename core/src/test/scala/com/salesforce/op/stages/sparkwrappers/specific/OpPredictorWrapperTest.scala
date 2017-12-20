/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.features.types._
import com.salesforce.op.test.{PrestigeData, TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class OpPredictorWrapperTest extends FlatSpec with TestSparkContext with PrestigeData {

  val log = LoggerFactory.getLogger(this.getClass)

  val (ds, targetLabel, featureVector) = TestFeatureBuilder[Real, OPVector](
    prestigeSeq.map(p => p.prestige.toReal -> Vectors.dense(p.education, p.income, p.women).toOPVector)
  )

  Spec[OpPredictorWrapper[_, _, _, _]] should
    "be able to run a simple logistic regression model (fitIntercept=true)" in {
    val lrModel: LinearRegressionModel = fitLinRegModel(fitIntercept = true)
    lrModel.intercept.abs should be > 1E-6
  }

  it should "be able to run a simple logistic regression model (fitIntercept=false)" in {
    val lrModel: LinearRegressionModel = fitLinRegModel(fitIntercept = false)
    lrModel.intercept.abs should be < Double.MinPositiveValue
  }

  private def fitLinRegModel(fitIntercept: Boolean): LinearRegressionModel = {
    val lrBase =
      new LinearRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)
        .setFitIntercept(fitIntercept)

    val lr =
      new OpPredictorWrapper[Real, Real, LinearRegression, LinearRegressionModel](lrBase)
        .setInput(targetLabel, featureVector)

    // Fit the model
    val model = lr.fit(ds)
    val lrModel = model.getSparkMlStage().get

    // Print the coefficients and intercept for linear regression
    log.info(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    log.info(s"numIterations: ${trainingSummary.totalIterations}")
    log.info(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    log.info(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    log.info(s"r2: ${trainingSummary.r2}")
    // checking r2 as a cheap way to make sure things are running as intended.
    assert(trainingSummary.r2 > 0.9)

    if (log.isInfoEnabled) {
      val output = lrModel.transform(ds)
      output.show(false)
    }

    lrModel
  }
}
