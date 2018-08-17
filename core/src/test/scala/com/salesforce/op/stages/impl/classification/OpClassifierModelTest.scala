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

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.features.types.{Prediction, RealNN}
import com.salesforce.op.stages.sparkwrappers.specific.SparkModelConverter._
import com.salesforce.op.test._
import com.salesforce.op.testkit._
import ml.dmlc.xgboost4j.scala.spark.{OpXGBoost, OpXGBoostQuietLogging, XGBoostClassifier}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalactic.Equality
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpClassifierModelTest extends FlatSpec with TestSparkContext with OpXGBoostQuietLogging {

  private val label = RandomIntegral.integrals(0, 2).limit(1000)
    .map{ v => RealNN(v.value.map(_.toDouble).getOrElse(0.0)) }
  private val fv = RandomVector.binary(10, 0.3).limit(1000)

  private val data = label.zip(fv)

  private val (rawDF, labelF, featureV) = TestFeatureBuilder("label", "features", data)

  Spec[OpDecisionTreeClassificationModel] should "produce the same values as the spark version" in {
    val spk = new DecisionTreeClassifier()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(spk, spk.uid).setInput(labelF, featureV)
    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpLogisticRegressionModel] should "produce the same values as the spark version" in {
    val spk = new LogisticRegression()
      .setFamily("multinomial")
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(spk, spk.uid).setInput(labelF, featureV)
    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpNaiveBayesModel] should "produce the same values as the spark version" in {
    val spk = new NaiveBayes()
      .setModelType("multinomial")
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(spk, uid = spk.uid).setInput(labelF, featureV)
    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpRandomForestClassificationModel] should "produce the same values as the spark version" in {
    val spk = new RandomForestClassifier()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(spk, spk.uid).setInput(labelF, featureV)
    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpGBTClassificationModel] should "produce the same values as the spark version" in {
    val spk = new GBTClassifier()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)

    val op = toOP(spk, spk.uid).setInput(labelF, featureV)
    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpLinearSVCModel] should "produce the same values as the spark version" in {
    val spk = new LinearSVC()
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)
    val op = toOP(spk, spk.uid).setInput(labelF, featureV)
    compareOutputsPred(spk.transform(rawDF), op.transform(rawDF), 3)
  }

  Spec[OpMultilayerPerceptronClassificationModel] should "produce the same values as the spark version" in {
    val spk = new MultilayerPerceptronClassifier()
      .setLayers(Array(10, 5, 4, 2)) // this is hard to generalize input layer must = number of features
      // output layer must equal number of labels
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
      .fit(rawDF)
    val op = toOP(spk, spk.uid).setInput(labelF, featureV)
    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  Spec[OpXGBoostClassifier] should "produce the same values as the spark version" in {
    val cl = new XGBoostClassifier()
    cl.set(cl.trackerConf, OpXGBoost.DefaultTrackerConf)
      .setFeaturesCol(featureV.name)
      .setLabelCol(labelF.name)
    val spk = cl.fit(rawDF)
    val op = toOP(spk, spk.uid).setInput(labelF, featureV)

    // ******************************************************
    // TODO: remove equality tolerance once XGBoost rounding bug in XGBoostClassifier.transform(probabilityUDF) is fixed
    // TODO: ETA - will be added in XGBoost version 0.81
    implicit val doubleEquality = new Equality[Double] {
      def areEqual(a: Double, b: Any): Boolean = b match {
        case s: Double => (a.isNaN && s.isNaN) || math.abs(a - s) < 0.0000001
        case _ => false
      }
    }
    implicit val doubleArrayEquality = new Equality[Array[Double]] {
      def areEqual(a: Array[Double], b: Any): Boolean = b match {
        case s: Array[_] if a.length == s.length => a.zip(s).forall(v => doubleEquality.areEqual(v._1, v._2))
        case _ => false
      }
    }
    // ******************************************************
    compareOutputs(spk.transform(rawDF), op.transform(rawDF))
  }

  def compareOutputs(df1: DataFrame, df2: DataFrame)(implicit arrayEquality: Equality[Array[Double]]): Unit = {
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

  def compareOutputsPred(df1: DataFrame, df2: DataFrame, predIndex: Int): Unit = {
    val sorted1 = df1.collect().sortBy(_.getAs[Double](predIndex))
    val sorted2 = df2.collect().sortBy(_.getAs[Map[String, Double]](2)(Prediction.Keys.PredictionName))
    sorted1.zip(sorted2).foreach{ case (r1, r2) =>
      val map = r2.getAs[Map[String, Double]](2)
      r1.getAs[Double](predIndex) shouldEqual map(Prediction.Keys.PredictionName)
    }
  }
}
