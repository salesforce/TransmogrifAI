/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.insights

import com.salesforce.op.FeatureHistory
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.preparators.{CorrelationType, SanityCheckDataTest}
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomReal, RandomVector}
import com.salesforce.op.utils.spark.{OpVectorColumnHistory, OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.utils.spark.RichDataset._

@RunWith(classOf[JUnitRunner])
class RecordInsightsCorrTest extends FlatSpec with TestSparkContext {

  Spec[RecordInsightsCorr] should "set parameters correctly" in {
    val insightsEstimator = new RecordInsightsCorr()
    insightsEstimator.getTopK shouldBe 20
    insightsEstimator.setTopK(2).getTopK shouldBe 2
    insightsEstimator.getCorrelationType shouldBe CorrelationType.Pearson
    insightsEstimator.setCorrelationType(CorrelationType.Spearman).getCorrelationType shouldBe CorrelationType.Spearman
    insightsEstimator.getNormType shouldBe NormType.MinMax
    insightsEstimator.setNormType(NormType.Znorm).getNormType shouldBe NormType.Znorm
  }


  it  should "work with randomly generated features and one prediction" in {
    val features = RandomVector.sparse(RandomReal.normal(), 40).limit(1000)
    val predicitons = RandomVector.dense(RandomReal.normal(), 1).limit(1000)
    val (df, f1, p1) = TestFeatureBuilder("features", "predictions", features.zip(predicitons))
    val dfWithMeta = addMetaData(df, "features", 40)
    val p1r = p1.copy(isResponse = true)

    val insightsEstimator = new RecordInsightsCorr().setInput(p1r, f1)
    val insights = insightsEstimator
      .fit(dfWithMeta)
      .transform(dfWithMeta)
      .collect(insightsEstimator.getOutput())
    insights.foreach(_.value.size shouldBe 20)
    val parsed = insights.map(RecordInsightsParser.parseInsights)
    parsed.foreach(_.values.foreach( i => i.foreach( v => v._1 shouldBe 0 )))
    parsed.foreach(_.values.foreach( i => i.foreach( v => math.abs(v._2) > 0 shouldBe true )))
  }

  it should "work with randomly generated features and multiple predictions" in {
    val features = RandomVector.dense(RandomReal.poisson(1), 10).limit(1000)
    val predicitons = RandomVector.dense(RandomReal.normal(), 5).limit(1000)
    val (df, f1, p1) = TestFeatureBuilder("features", "predictions", features.zip(predicitons))
    val dfWithMeta = addMetaData(df, "features", 10)
    val p1r = p1.copy(isResponse = true)

    val insightsEstimator = new RecordInsightsCorr().setInput(p1r, f1).setTopK(2)
    val insights = insightsEstimator
      .fit(dfWithMeta)
      .transform(dfWithMeta)
      .collect(insightsEstimator.getOutput())
      .map(RecordInsightsParser.parseInsights)
    insights.map( _.count{ case (k, v) => v.exists(_._1 == 4) } shouldBe 2 )
  }

  private def addMetaData(df: DataFrame, fieldName: String, size: Int): DataFrame = {
    val columns = (0 until size).map(_.toString).map(i => new OpVectorColumnMetadata(Seq(i), Seq(i), Some(i), Some(i)))
    val hist = (0 until size).map(_.toString).map(i => i -> FeatureHistory(Seq(s"a_$i"), Seq(s"b_$i"))).toMap
    val metadata = OpVectorMetadata(fieldName, columns.toArray, hist).toMetadata
    val fields = df.schema.fields.map{ f =>
      if (f.name == fieldName) f.copy(metadata = metadata)
      else f
    }
    spark.createDataFrame(df.rdd, StructType(fields))
  }

  // scalastyle:off
  private val data = Seq(
    SanityCheckDataTest("a",  32,  5.0,   0,  0.9,  0.5,  0),
    SanityCheckDataTest("b",  32,  4.0,  -1,  0.1,  0,  0.1),
    SanityCheckDataTest("a",  32,  6.0,   0,  0.8,  0.5,  0),
    SanityCheckDataTest("a",  32,  5.5,   0,  0.85, 0.5,  0),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.05, 0,  0.1),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.2,  0,  0.1),
    SanityCheckDataTest("a",  32,  5.0,   0,  0.99, 0.5,  0),
    SanityCheckDataTest("b",  32,  4.0,   0,  0.0,  0,  0.1),
    SanityCheckDataTest("a",  32,  6.0,  -1,  0.7,  0.5,  0),
    SanityCheckDataTest("a",  32,  5.5,   0,  0.8,  0.5,  0),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.1,  0,  0.1),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.05, 0,  0.1),
    SanityCheckDataTest("a",  32,  5.0,   0,  1,    0.5,  0),
    SanityCheckDataTest("b",  32,  4.0,  -1,  0.1,  0,  0.1),
    SanityCheckDataTest("a",  32,  6.0,  -1,  0.9,  0.5,  0),
    SanityCheckDataTest("a",  32,  5.5,   0,  1,    0.5,  0),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.2,  0,  0.1),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.3,  0,  0.1),
    SanityCheckDataTest("a",  32,  5.0,   0,  0.6,  0.5,  0),
    SanityCheckDataTest("b",  32,  4.0,  -1,  0.1,  0,  0.1),
    SanityCheckDataTest("a",  32,  6.0,   0,  0.9,  0.5,  0),
    SanityCheckDataTest("a",  32,  5.5,   0,  1,    0.5,  0),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.05, 0,  0.1),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.3,  0,  0.1),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.05, 0,  0.1),
    SanityCheckDataTest("b",  32,  5.4,  -1,  0.4,  0,  0.1)
  ).map(data =>
    (
      data.name.toText,
      Seq(data.isBlueEyed, 1 - data.isBlueEyed).toOPVector,
      Seq(data.age, data.height, data.height_null, data.gender, data.testFeatNegCor).toOPVector
    )
  )
  // scalastyle:on

  private val (testData, name, predictionNoRes, featureVector) = TestFeatureBuilder("name", "pred", "features", data)
  private val prediction = predictionNoRes.copy(isResponse = true)
  private val testDataMeta = addMetaData(testData, "features", 5)

  private val estimator = new RecordInsightsCorr().setInput(prediction, featureVector)

  private def parse(df: DataFrame) = df.collect(name, estimator.getOutput())
    .map { case (n, i) => n -> RecordInsightsParser.parseInsights(i) }

  it should "return the most predictive features" in {
    val insights = estimator.setTopK(1).fit(testDataMeta).transform(testDataMeta)
    val parsed = parse(insights)
    // the highest corr that value that is not zero should be the top feature
    parsed.foreach{ case (k, in) =>
      if (k.value.contains("a")) in.head._1.columnName shouldBe "3_3_3_3"
      else in.head._1.columnName shouldBe "4_4_4_4"
    }
    // the scores should be the same but opposite in sign
    parsed.foreach{ case (_, in) => math.abs(in.head._2(0)._2 + in.head._2(1)._2) < 0.00001 shouldBe true }
  }

  private def haveDiffScores(
    parsed1: Array[(Text, Map[OpVectorColumnHistory, Seq[(Int, Double)]])],
    parsed2: Array[(Text, Map[OpVectorColumnHistory, Seq[(Int, Double)]])]
  ) = parsed1.zip(parsed2)
    .foreach{ case ((k1, in1), (k2, in2)) =>
    k1 shouldEqual k2
    in1.keys.map( k => in2(k).sortBy(_._1).zip(in1(k).sortBy(_._1)).map{
      case ((_, v1), (_, v2)) => if (math.abs(v1) > 0.00001) math.abs(v1 - v2) > 0.00001 shouldBe true
    })
  }

  it should "have different values when different norms" in {
    estimator.setTopK(5)

    estimator.setNormType(NormType.Znorm)
    val insights0 = estimator.fit(testDataMeta).transform(testDataMeta)
    val parsed0 = parse(insights0)

    estimator.setNormType(NormType.MinMax)
    val insights1 = estimator.fit(testDataMeta).transform(testDataMeta)
    val parsed1 = parse(insights1)

    estimator.setNormType(NormType.MinMaxCentered)
    val insights2 = estimator.fit(testDataMeta).transform(testDataMeta)
    val parsed2 = parse(insights2)

    haveDiffScores(parsed0, parsed1)
    haveDiffScores(parsed0, parsed2)
  }

  it should "have different values when different correlations are used" in {
    estimator.setCorrelationType(CorrelationType.Spearman)
    val insights0 = estimator.fit(testDataMeta).transform(testDataMeta)
    val parsed0 = parse(insights0)

    estimator.setCorrelationType(CorrelationType.Pearson)
    val insights1 = estimator.fit(testDataMeta).transform(testDataMeta)
    val parsed1 = parse(insights1)

    haveDiffScores(parsed0, parsed1)
  }
}
