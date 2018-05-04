/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */
package org.apache.spark.ml.regression

import com.salesforce.op.features.types.{Prediction, RealNN}
import com.salesforce.op.test._
import com.salesforce.op.testkit._
import org.apache.spark.ml.SparkModelConverter.toOP
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpPredictionModelTest extends FlatSpec with TestSparkContext {

  private val label = RandomIntegral.integrals(0, 2).limit(1000)
    .map{ v => RealNN(v.value.map(_.toDouble).getOrElse(0.0)) }
  private val fv = RandomVector.binary(10, 0.3).limit(1000)

  private val data = label.zip(fv)

  private val (rawDF, labelF, featureV) = TestFeatureBuilder("label", "features", data)

  Spec[OpDecisionTreeRegressionModel] should "produce the same values as the spark version" in {
    val spk = new DecisionTreeRegressor()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(Some(spk)).setInput(labelF, featureV)

    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }


  Spec[OpLinearPredictionModel] should "produce the same values as the spark version" in {
    val spk = new LinearRegression()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(Some(spk)).setInput(labelF, featureV)

    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpGBTRegressionModel] should "produce the same values as the spark version" in {
    val spk = new GBTRegressor()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(Some(spk)).setInput(labelF, featureV)

    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpRandomForestRegressionModel] should "produce the same values as the spark version" in {
    val spk = new RandomForestRegressor()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(Some(spk)).setInput(labelF, featureV)

    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }


  def compareOutputs(df1: DataFrame, df2: DataFrame): Unit = {
    val sorted1 = df1.collect().sortBy(_.getAs[Double](2))
    val sorted2 = df2.collect().sortBy(_.getAs[Map[String, Double]](2)(Prediction.Keys.PredictionName))
    sorted1.zip(sorted2).foreach{ case (r1, r2) =>
      val map = r2.getAs[Map[String, Double]](2)
      r1.getAs[Double](2) shouldEqual map(Prediction.Keys.PredictionName)
    }
  }
}
