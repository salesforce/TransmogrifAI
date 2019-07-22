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

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.PredictionEquality
import com.salesforce.op.stages.impl.classification.QuantileRegressionRF
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpPredictorWrapperModel}
import com.salesforce.op.test._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.mllib.random.RandomRDDs._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.apache.spark.sql.functions._


import scala.util.Random

@RunWith(classOf[JUnitRunner])
class OpRandomForestRegressorTest extends OpEstimatorSpec[Prediction,
  OpPredictorWrapperModel[RandomForestRegressionModel],
  OpPredictorWrapper[RandomForestRegressor, RandomForestRegressionModel]] with PredictionEquality {


  override def specName: String = Spec[OpRandomForestRegressor]

  val (inputData, rawLabel, features) = TestFeatureBuilder("labels", "features",
    Seq[(RealNN, OPVector)](
      (10.0.toRealNN, Vectors.dense(1.0, 4.3, 1.3).toOPVector),
      (20.0.toRealNN, Vectors.dense(2.0, 0.3, 0.1).toOPVector),
      (30.0.toRealNN, Vectors.dense(3.0, 3.9, 4.3).toOPVector),
      (40.0.toRealNN, Vectors.dense(4.0, 1.3, 0.9).toOPVector),
      (50.0.toRealNN, Vectors.dense(5.0, 4.7, 1.3).toOPVector)
    )
  )
  val label = rawLabel.copy(isResponse = true)
  val estimator = new OpRandomForestRegressor().setInput(label, features)

  val expectedResult = Seq(
    Prediction(20.0),
    Prediction(23.5),
    Prediction(31.5),
    Prediction(35.5),
    Prediction(37.0)
  )

  val (seed, smallCount, bigCount) = (1234L, 20, 1000)


  import spark.implicits._



  it should "allow the user to set the desired spark parameters" in {
    estimator
      .setMaxDepth(7)
      .setMaxBins(3)
      .setMinInstancesPerNode(2)
      .setMinInfoGain(0.1)
      .setSeed(42L)
    val model = estimator.fit(inputData)

    estimator.predictor.getMaxDepth shouldBe 7
    estimator.predictor.getMaxBins shouldBe 3
    estimator.predictor.getMinInstancesPerNode shouldBe 2
    estimator.predictor.getMinInfoGain shouldBe 0.1
    estimator.predictor.getSeed shouldBe 42L

    val trees = model.getSparkMlStage().get.trees

    val quantileRegressionRF = new QuantileRegressionRF(trees = trees).setInput(label, features)
      .setPercentageLevel(0.90)


    val transformedData = model.transform(inputData)
    quantileRegressionRF.fit(transformedData).transform(transformedData).show()

  }

  it should "QRF with range" in {
    val df =
      normalVectorRDD(sc, bigCount, 3, seed = seed)
        .map(v => Random.nextDouble() * 10000.0 -> Vectors.dense(v.toArray))


    val outliers =
      normalVectorRDD(sc, 10, 3, seed = seed)
        .map(v => Random.nextDouble() * 1000000.0 -> Vectors.dense(v.toArray.map(_ * 1000 + 500)))

    val train = df.union(outliers).toDF("labels", "features")

    train.show()


    val holdout = normalVectorRDD(sc, smallCount, 3, seed = seed)
      .map(v => 0.0 -> Vectors.dense(v.toArray.map(_ * 1000 + 500))).toDF("labels", "features")


    val rf = new OpRandomForestRegressor().setInput(label, features)
    rf
      .setMaxDepth(7)
      .setMaxBins(3)
      .setMinInstancesPerNode(2)
      .setMinInfoGain(0.1)
      .setSeed(42L)
    val model = rf.fit(train)


    val trees = model.getSparkMlStage().get.trees

    val quantileRegressionRF = new QuantileRegressionRF(trees = trees).setInput(label, features)
      .setPercentageLevel(0.90)


    val transformedTrain = model.transform(train)
    val transformedHoldout = model.transform(holdout)
    val transformed = quantileRegressionRF.fit(transformedTrain).transform(transformedHoldout)


    val getL = (m: RealMap#Value) => m("lowerQuantile")
    val getU = (m: RealMap#Value) => m("upperQuantile")
    val getPred = (m: Prediction#Value) => m("prediction")
    val lUDF = udf(getL)
    val uUDF = udf(getU)
    val predUDF = udf(getPred)
    val output = col(quantileRegressionRF.getOutput().name)
    transformed.select(predUDF(col(model.getOutput().name)), lUDF(output), uUDF(output)).show()

  }


  it should "QRF with lower range" in {

    val df =
      normalVectorRDD(sc, bigCount, 3, seed = seed)
        .map(v => Random.nextDouble() * 1000.0 -> Vectors.dense(v.toArray))


    val outliers =
      normalVectorRDD(sc, 1, 3, seed = seed)
        .map(v => Random.nextDouble() / 10000.0 - 1000000 -> Vectors.dense(v.toArray.map(_ / 1000 - 500)))

    val train = df.union(outliers).toDF("labels", "features")

    train.show()


    val holdout = normalVectorRDD(sc, smallCount, 3, seed = seed)
      .map(v => 0.0 -> Vectors.dense(v.toArray.map(_ / 1000 - 500))).toDF("labels", "features")


    val rf = new OpRandomForestRegressor().setInput(label, features)
    rf
      .setMaxDepth(7)
      .setMaxBins(3)
      .setMinInstancesPerNode(2)
      .setMinInfoGain(0.1)
      .setSeed(42L)
    val model = rf.fit(train)


    val trees = model.getSparkMlStage().get.trees

    val quantileRegressionRF = new QuantileRegressionRF(trees = trees).setInput(label, features)
      .setPercentageLevel(0.90)


    val transformedTrain = model.transform(train)
    val transformedHoldout = model.transform(holdout)
    val transformed = quantileRegressionRF.fit(transformedTrain).transform(transformedHoldout)


    val getL = (m: RealMap#Value) => m("lowerQuantile")
    val getU = (m: RealMap#Value) => m("upperQuantile")
    val getPred = (m: Prediction#Value) => m("prediction")
    val lUDF = udf(getL)
    val uUDF = udf(getU)
    val predUDF = udf(getPred)
    val output = col(quantileRegressionRF.getOutput().name)
    transformed.select(predUDF(col(model.getOutput().name)), lUDF(output), uUDF(output)).show()

  }
}
