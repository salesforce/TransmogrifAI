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

package com.salesforce.op.stages.impl.insights

import com.salesforce.op.FeatureHistory
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.{OpLogisticRegression, OpRandomForestClassifier}
import com.salesforce.op.stages.impl.preparators.SanityCheckDataTest
import com.salesforce.op.stages.impl.regression.OpLinearRegression
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomIntegral, RandomReal, RandomVector}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.classification.{LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class RecordInsightsLOCOTest extends FlatSpec with TestSparkContext {

  Spec[RecordInsightsLOCO[_]] should "work with randomly generated features and binary logistic regression" in {
    val features = RandomVector.sparse(RandomReal.normal(), 40).limit(1000)
    val labels = RandomIntegral.integrals(0, 2).limit(1000).map(_.value.get.toRealNN)
    val (df, f1, l1) = TestFeatureBuilder("features", "labels", features.zip(labels))
    val l1r = l1.copy(isResponse = true)
    val dfWithMeta = addMetaData(df, "features", 40)
    val sparkModel = new OpLogisticRegression().setInput(l1r, f1).fit(df)

    val model = sparkModel.asInstanceOf[SparkWrapperParams[_]].getSparkMlStage().get
      .asInstanceOf[LogisticRegressionModel]

    // val model = sparkModel.getSparkMlStage().get
    val insightsTransformer = new RecordInsightsLOCO(model).setInput(f1)
    val insights = insightsTransformer.transform(dfWithMeta).collect(insightsTransformer.getOutput())
    insights.foreach(_.value.size shouldBe 20)
    val parsed = insights.map(RecordInsightsParser.parseInsights)
    parsed.map( _.count{ case (_, v) => v.exists(_._1 == 1) } shouldBe 20 ) // number insights per pred column
    parsed.foreach(_.values.foreach(i => i.foreach(v => math.abs(v._2) > 0 shouldBe true)))
  }

  it should "work with randomly generated features and multiclass random forest" in {
    val features = RandomVector.sparse(RandomReal.normal(), 40).limit(1000)
    val labels = RandomIntegral.integrals(0, 5).limit(1000).map(_.value.get.toRealNN)
    val (df, f1, l1) = TestFeatureBuilder("features", "labels", features.zip(labels))
    val l1r = l1.copy(isResponse = true)
    val dfWithMeta = addMetaData(df, "features", 40)
    val sparkModel = new OpRandomForestClassifier().setInput(l1r, f1).fit(df)
    val model = sparkModel.asInstanceOf[SparkWrapperParams[_]].getSparkMlStage().get
      .asInstanceOf[RandomForestClassificationModel]

    val insightsTransformer = new RecordInsightsLOCO(model).setInput(f1).setTopK(2)
    val insights = insightsTransformer.transform(dfWithMeta).collect(insightsTransformer.getOutput())
    insights.foreach(_.value.size shouldBe 2)
    val parsed = insights.map(RecordInsightsParser.parseInsights)
    parsed.map( _.count{ case (_, v) => v.exists(_._1 == 5) } shouldBe 0 ) // no 6th column of insights
    parsed.map( _.count{ case (_, v) => v.exists(_._1 == 4) } shouldBe 2 ) // number insights per pred column
    parsed.map( _.count{ case (_, v) => v.exists(_._1 == 3) } shouldBe 2 ) // number insights per pred column
    parsed.map( _.count{ case (_, v) => v.exists(_._1 == 2) } shouldBe 2 ) // number insights per pred column
    parsed.map( _.count{ case (_, v) => v.exists(_._1 == 1) } shouldBe 2 ) // number insights per pred column
    parsed.map( _.count{ case (_, v) => v.exists(_._1 == 0) } shouldBe 2 ) // number insights per pred column
  }


  it should "work with randomly generated features and linear regression" in {
    val features = RandomVector.sparse(RandomReal.normal(), 40).limit(1000)
    val labels = RandomReal.normal[RealNN]().limit(1000)
    val (df, f1, l1) = TestFeatureBuilder("features", "labels", features.zip(labels))
    val l1r = l1.copy(isResponse = true)
    val dfWithMeta = addMetaData(df, "features", 40)
    val sparkModel = new OpLinearRegression().setInput(l1r, f1).fit(df)
    val model = sparkModel.asInstanceOf[SparkWrapperParams[_]].getSparkMlStage().get
      .asInstanceOf[LinearRegressionModel]

    val insightsTransformer = new RecordInsightsLOCO(model).setInput(f1)
    val insights = insightsTransformer.transform(dfWithMeta).collect(insightsTransformer.getOutput())
    insights.foreach(_.value.size shouldBe 20)
    val parsed = insights.map(RecordInsightsParser.parseInsights)
    parsed.foreach(_.values.foreach(i => i.foreach(v => v._1 shouldBe 0))) // has only one pred column
    parsed.foreach(_.values.foreach(i => i.foreach(v => math.abs(v._2) > 0 shouldBe true)))
  }

  private def addMetaData(df: DataFrame, fieldName: String, size: Int): DataFrame = {
    val columns = (0 until size).map(_.toString).map(i => new OpVectorColumnMetadata(Seq(i), Seq(i), Some(i), Some(i)))
    val hist = (0 until size).map(_.toString).map(i => i -> FeatureHistory(Seq(s"a_$i"), Seq(s"b_$i"))).toMap
    val metadata = OpVectorMetadata(fieldName, columns.toArray, hist).toMetadata
    val fields = df.schema.fields.map { f =>
      if (f.name == fieldName) f.copy(metadata = metadata)
      else f
    }
    spark.createDataFrame(df.rdd, StructType(fields))
  }

  it should "return the most predictive features" in {

    // scalastyle:off
    val data = Seq( // name, age, height, height_null, isBlueEyed, gender, testFeatNegCor
      SanityCheckDataTest("a", 32, 5.0, 0, 0.9, 0.5, 0),
      SanityCheckDataTest("b", 32, 4.0, 1, 0.1, 0, 0.1),
      SanityCheckDataTest("a", 32, 6.0, 0, 0.8, 0.5, 0),
      SanityCheckDataTest("a", 32, 5.5, 0, 0.85, 0.5, 0),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.05, 0, 0.1),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.2, 0, 0.1),
      SanityCheckDataTest("a", 32, 5.0, 0, 0.99, 0.5, 0),
      SanityCheckDataTest("b", 32, 4.0, 0, 0.0, 0, 0.1),
      SanityCheckDataTest("a", 32, 6.0, 1, 0.7, 0.5, 0),
      SanityCheckDataTest("a", 32, 5.5, 0, 0.8, 0.5, 0),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.1, 0, 0.1),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.05, 0, 0.1),
      SanityCheckDataTest("a", 32, 5.0, 0, 1, 0.5, 0),
      SanityCheckDataTest("b", 32, 4.0, 1, 0.1, 0, 0.1),
      SanityCheckDataTest("a", 32, 6.0, 1, 0.9, 0.5, 0),
      SanityCheckDataTest("a", 32, 5.5, 0, 1, 0.5, 0),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.2, 0, 0.1),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.3, 0, 0.1),
      SanityCheckDataTest("a", 32, 5.0, 0, 0.6, 0.5, 0),
      SanityCheckDataTest("b", 32, 4.0, 1, 0.1, 0, 0.1),
      SanityCheckDataTest("a", 32, 6.0, 0, 0.9, 0.5, 0),
      SanityCheckDataTest("a", 32, 5.5, 0, 1, 0.5, 0),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.05, 0, 0.1),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.3, 0, 0.1),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.05, 0, 0.1),
      SanityCheckDataTest("b", 32, 5.4, 1, 0.4, 0, 0.1)
    ).map(data =>
      (
        data.name.toText,
        data.height_null.toRealNN,
        Seq(data.age, data.height, data.isBlueEyed, data.gender, data.testFeatNegCor).toOPVector
      )
    )
    // scalastyle:on

    val (testData, name, labelNoRes, featureVector) = TestFeatureBuilder("name", "label", "features", data)
    val label = labelNoRes.copy(isResponse = true)
    val testDataMeta = addMetaData(testData, "features", 5)
    val sparkModel = new OpLogisticRegression().setInput(label, featureVector).fit(testData)
    val model = sparkModel.asInstanceOf[SparkWrapperParams[_]].getSparkMlStage().get
      .asInstanceOf[LogisticRegressionModel]

    val transformer = new RecordInsightsLOCO(model).setInput(featureVector)

    val insights = transformer.setTopK(1).transform(testDataMeta)
    val parsed = insights.collect(name, transformer.getOutput())
      .map { case (n, i) => n -> RecordInsightsParser.parseInsights(i) }
    // the highest corr that value that is not zero should be the top feature
    parsed.foreach { case (_, in) => Set("3_3_3_3", "1_1_1_1").contains(in.head._1.columnName) shouldBe true }
    // the scores should be the same but opposite in sign
    parsed.foreach { case (_, in) => math.abs(in.head._2(0)._2 + in.head._2(1)._2) < 0.00001 shouldBe true }
  }

}
