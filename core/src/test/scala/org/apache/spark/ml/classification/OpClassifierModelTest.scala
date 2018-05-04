/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */
package org.apache.spark.ml.classification

import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.test._
import com.salesforce.op.testkit._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.apache.spark.ml.SparkModelConverter._
import org.apache.spark.ml.linalg.Vector



@RunWith(classOf[JUnitRunner])
class OpClassifierModelTest extends FlatSpec with TestSparkContext {

  private val label = RandomIntegral.integrals(0, 2).limit(1000)
    .map{ v => RealNN(v.value.map(_.toDouble).getOrElse(0.0)) }
  private val fv = RandomVector.binary(10, 0.3).limit(1000)

  private val data = label.zip(fv)

  private val (rawDF, labelF, featureV) =
    TestFeatureBuilder("label", "features", data)


  Spec[OpDecisionTreeClassificationModel] should "produce the same values as the spark version" in {
    val spk = new DecisionTreeClassifier()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(Some(spk)).setInput(labelF, featureV)

    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }


  Spec[OpLogisticRegressionModel] should "produce the same values as the spark version" in {
    val spk = new LogisticRegression()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(Some(spk)).setInput(labelF, featureV)

    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpNaiveBayesModel] should "produce the same values as the spark version" in {
    val spk = new NaiveBayes()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(Some(spk), isMultinomial = true).setInput(labelF, featureV)

    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpRandomForestClassificationModel] should "produce the same values as the spark version" in {
    val spk = new RandomForestClassifier()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(Some(spk)).setInput(labelF, featureV)

    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  def compareOutputs(df1: DataFrame, df2: DataFrame): Unit = {

    def keysStartsWith(name: String, value: Map[String, Double]): Array[Double] = {
      val names = value.keys.filter(_.startsWith(name)).toArray.sorted
      names.map(value)
    }
    val sorted1 = df1.collect().sortBy(_.getAs[Double](4))
    val sorted2 = df2.collect().sortBy(_.getAs[Map[String, Double]](2)(Prediction.Keys.PredictionName))
    sorted1.zip(sorted2).foreach{ case (r1, r2) =>
      val map = r2.getAs[Map[String, Double]](2)
      r1.getAs[Double](4) shouldEqual map(Prediction.Keys.PredictionName)
      r1.getAs[Vector](3).toArray shouldEqual keysStartsWith(Prediction.Keys.ProbabilityName, map)
      r1.getAs[Vector](2).toArray shouldEqual keysStartsWith(Prediction.Keys.RawPredictionName, map)
    }
  }
}
