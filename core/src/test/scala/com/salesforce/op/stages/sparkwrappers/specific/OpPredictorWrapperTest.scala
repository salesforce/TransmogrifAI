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

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.features.types._
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
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

  val (ds, targetLabel, featureVector) = TestFeatureBuilder[RealNN, OPVector](
    prestigeSeq.map(p => p.prestige.toRealNN -> Vectors.dense(p.education, p.income, p.women).toOPVector)
  )

  Spec[OpPredictorWrapper[_, _]] should
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

    val lr = new OpPredictorWrapper[LinearRegression, LinearRegressionModel](lrBase)
      .setInput(targetLabel, featureVector)

    // Fit the model
    val model = lr.fit(ds).asInstanceOf[SparkWrapperParams[LinearRegressionModel]]
    val lrModel = model.getSparkMlStage().get

    // Print the coefficients and intercept for linear regression
    log.info(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    log.info(s"numIterations: ${trainingSummary.totalIterations}")
    log.info(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    if (log.isInfoEnabled) trainingSummary.residuals.show()
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
