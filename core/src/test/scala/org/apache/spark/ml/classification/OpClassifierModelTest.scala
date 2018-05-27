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
