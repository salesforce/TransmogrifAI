/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.{CustomReader, DataReaders}
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.{OpWorkflow, _}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{sum, udf}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FlatSpec


@RunWith(classOf[JUnitRunner])
class OpStandardScalerTest extends FlatSpec with TestSparkContext {
  import spark.implicits._

  // TODO: use TestFeatureBuilder instead
  lazy val testData: Dataset[StdScTestData] = DataStdScTest.input.toDS()

  // create the feature to which the normalizer transformer will be applied
  val someNumericFeature = FeatureBuilder.RealNN[StdScTestData].extract(_.someNumericFeature.toRealNN).asPredictor

  Spec[OpScalarStandardScaler] should
    "scale scalars properly outside a pipeline (by itself); setWithMean(false).setWithStd(false)" in {

    val featureNormalizer = new OpScalarStandardScaler().setWithMean(false).setWithStd(false)

    // apply the normalizer to the desired feature column
    val scalerModel = featureNormalizer.setInput(someNumericFeature).fit(testData)
    val normalizedFeatureDF = scalerModel.setInput(someNumericFeature).transform(testData)

    val sumSqDist =
      validateDataframeDoubleColumn(normalizedFeatureDF, scalerModel.getOutput().name, "someNumericFeature")

    assert(sumSqDist <= 0.000001, "===> the sum of squared distances between actual and expected should be zero.")
  }

  it should "scale scalars properly outside a pipeline (by itself); setWithMean(false).setWithStd(true)" in {
    val featureNormalizer = new OpScalarStandardScaler().setWithMean(false).setWithStd(true)

    // apply the normalizer to the desired feature column
    val scalerModel = featureNormalizer.setInput(someNumericFeature).fit(testData)
    val normalizedFeatureDF = scalerModel.setInput(someNumericFeature).transform(testData)

    val sumSqDist =
      validateDataframeDoubleColumn(normalizedFeatureDF, scalerModel.getOutput().name, "normalizedWithStdDevOnly")

    assert(sumSqDist <= 0.000001, "===> the sum of squared distances between actual and expected should be zero.")
  }

  it should "scale scalars properly outside a pipeline (by itself); setWithMean(true).setWithStd(false)" in {
    val featureNormalizer = new OpScalarStandardScaler().setWithMean(true).setWithStd(false)

    // apply the normalizer to the desired feature column
    val scalerModel = featureNormalizer.setInput(someNumericFeature).fit(testData)
    val normalizedFeatureDF = scalerModel.setInput(someNumericFeature).transform(testData)

    val sumSqDist =
      validateDataframeDoubleColumn(normalizedFeatureDF, scalerModel.getOutput().name, "normalizedWithMeanOnly")

    assert(sumSqDist <= 0.000001, "===> the sum of squared distances between actual and expected should be zero.")
  }

  it should "scale scalars properly outside a pipeline (by itself); basic test with defaults for stddev & mean" in {
    val featureNormalizer = new OpScalarStandardScaler()

    // apply the normalizer to the desired feature column
    val scalerModel = featureNormalizer.setInput(someNumericFeature).fit(testData)
    val normalizedFeatureDF = scalerModel.setInput(someNumericFeature).transform(testData)

    val sumSqDist = validateDataframeDoubleColumn(normalizedFeatureDF,
      scalerModel.getOutput().name, "expectedNormalizedScalarFeature")

    assert(sumSqDist <= 0.000001, "===> the sum of squared distances between actual and expected should be zero.")
  }

  it should "scale scalars properly with a shortcut (also integration test in a workflow)" in {
    val expectedNormalizedScalarFeature = FeatureBuilder.RealNN[StdScTestData]
      .extract(_.expectedNormalizedScalarFeature.toRealNN)
      .asPredictor

    // try the feature transformation shortcut
    val normedOutput = someNumericFeature.zNormalize()

    // create a pipeline and set the input features to be ingested by the pipeline.
    val workflow = new OpWorkflow().setResultFeatures(normedOutput, expectedNormalizedScalarFeature)

    val reader = new CustomReader[StdScTestData](DataReaders.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[StdScTestData], Dataset[StdScTestData]] =
        Left(spark.sparkContext.parallelize(DataStdScTest.input))
    }

    val workflowModel = workflow.setReader(reader).train()
    val normalizedFeatureDF = workflowModel.score()

    val sumSqDist = validateDataframeDoubleColumn(normalizedFeatureDF, normedOutput.name,
      "expectedNormalizedScalarFeature")

    assert(sumSqDist <= 0.000001, "===> the sum of squared distances between actual and expected should be zero.")
  }

  private def validateDataframeDoubleColumn(normalizedFeatureDF: DataFrame, scaledFeatureName: String,
    targetColumnName: String): Double = {
    val sqDistUdf = udf { (leftCol: Double, rightCol: Double) => Math.pow(leftCol - rightCol, 2) }

    // compute sum of squared distances between expected and actual
    val finalDF = normalizedFeatureDF.withColumn("sqDist",
      sqDistUdf(normalizedFeatureDF(scaledFeatureName), normalizedFeatureDF(targetColumnName)))

    val sumSqDist: Double = finalDF.agg(sum(finalDF("sqDist"))).first().getDouble(0)
    sumSqDist
  }
}


case class StdScTestData
(
  id: Double,
  featuresCol: Vector,
  featureCol: Vector,
  expectedNormalizedFeature: Vector,
  someNumericFeature: Double,
  expectedNormalizedScalarFeature: Double,
  normalizedWithMeanOnly: Double,
  normalizedWithStdDevOnly: Double
)

object DataStdScTest {
  // common dataset to be used in the tests
  val input = Seq(
    StdScTestData(0,
      Vectors.dense(1.0, 0.5, -1.0),
      Vectors.dense(1.0),
      Vectors.dense(-0.8728715609439697),
      1.0,
      -0.8728715609439697,
      -1.333333333333333,
      0.6546536707079772),
    StdScTestData(1,
      Vectors.dense(2.0, 1.0, 1.0),
      Vectors.dense(2.0),
      Vectors.dense(-0.2182178902359925),
      2.0,
      -0.2182178902359925,
      -0.33333333333333304,
      1.3093073414159544),
    StdScTestData(2,
      Vectors.dense(4.0, 10.0, 2.0),
      Vectors.dense(4.0),
      Vectors.dense(1.0910894511799618),
      4.0,
      1.0910894511799618,
      1.666666666666667,
      2.618614682831909)
  )
}
