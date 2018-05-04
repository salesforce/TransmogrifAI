/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.features.Feature
import com.salesforce.op.stages.base.unary.UnaryModel
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.{Estimator, Transformer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FlatSpec

import scala.util.Random

@RunWith(classOf[JUnitRunner])
class PercentileCalibratorTest extends FlatSpec with TestSparkContext {
  import spark.implicits._

  Spec[PercentileCalibrator] should "return a minimum calibrated score of 0 and max of 99 when buckets is 100" in {
    val data = (0 until 1000).map(i => i.toLong.toIntegral -> Random.nextDouble.toRealNN)
    val (scoresDF, f1, f2): (DataFrame, Feature[Integral], Feature[RealNN]) = TestFeatureBuilder(data)
    val percentile = f2.toPercentile()

    val model = percentile.originStage.asInstanceOf[Estimator[_]].fit(scoresDF)
    val scoresTransformed = model.asInstanceOf[Transformer].transform(scoresDF)

    percentile.name shouldBe percentile.originStage.getOutputFeatureName
    scoresTransformed.select(min(percentile.name)).first.getDouble(0) should equal (0.0)
    scoresTransformed.select(max(percentile.name)).first.getDouble(0) should equal (99.0)
  }

  it should "produce the calibration map metadata" in {
    Random.setSeed(123)
    val data = (0 until 3).map(i => i.toLong.toIntegral -> Random.nextDouble.toRealNN)
    val (scoresDF, f1, f2): (DataFrame, Feature[Integral], Feature[RealNN]) = TestFeatureBuilder(data)
    val percentile = f2.toPercentile()
    val model = percentile.originStage.asInstanceOf[Estimator[_]].fit(scoresDF)
    val trans = model.asInstanceOf[UnaryModel[_, _]]
    val splits = trans.getMetadata().getSummaryMetadata().getStringArray(PercentileCalibrator.OrigSplitsKey)
    val scaled = trans.getMetadata().getSummaryMetadata().getStringArray(PercentileCalibrator.ScaledSplitsKey)

    splits should contain theSameElementsAs
      Array(Double.NegativeInfinity, 0.7231742029971469, 0.9908988967772393, Double.PositiveInfinity).map(_.toString)
    scaled should contain theSameElementsAs Array(0.0, 50.0, 99.0, 99.0).map(_.toString)
  }

  it should "return a maximum calibrated score of 99" in {
    val data = (0 until 1000).map(i => i.toLong.toIntegral -> Random.nextDouble.toRealNN)
    val (scoresDF, f1, f2): (DataFrame, Feature[Integral], Feature[RealNN]) = TestFeatureBuilder(data)
    val percentile = f2.toPercentile()

    val model = percentile.originStage.asInstanceOf[Estimator[_]].fit(scoresDF)
    val scoresTransformed = model.asInstanceOf[Transformer].transform(scoresDF)

    scoresTransformed.select(max(percentile.name)).first.getDouble(0) should equal (99.0)
  }

  it should "return a maximum calibrated score of 99 when calibrating with less than 100" in {
    val data = (0 until 30).map(i => i.toLong.toIntegral -> Random.nextDouble.toRealNN)
    val (scoresDF, f1, f2): (DataFrame, Feature[Integral], Feature[RealNN]) = TestFeatureBuilder(data)
    val percentile = f2.toPercentile()

    val model = percentile.originStage.asInstanceOf[Estimator[_]].fit(scoresDF)
    val scoresTransformed = model.asInstanceOf[Transformer].transform(scoresDF)

    scoresTransformed.select(max(percentile.name)).first.getDouble(0) should equal (99.0)
  }

  it should "return all scores from 0 to 99 in increments of 1" in {
    val data = (0 until 1000).map(i => i.toLong.toIntegral -> Random.nextDouble.toRealNN)
    val (scoresDF, f1, f2): (DataFrame, Feature[Integral], Feature[RealNN]) = TestFeatureBuilder(data)
    val percentile = f2.toPercentile()

    val model = percentile.originStage.asInstanceOf[Estimator[_]].fit(scoresDF)
    val scoresTransformed = model.asInstanceOf[Transformer].transform(scoresDF)

    val scoreCounts = scoresTransformed.groupBy(percentile.name)
      .agg(count(percentile.name))
      .withColumnRenamed(s"count(${percentile.name})", "count")

    val checkSet = (0 to 99).map(_.toReal).toSet

    scoreCounts.collect(percentile).toSet should equal (checkSet)
  }

  it should "return a uniform distribution of scores" in {
    val data = (0 until 1000).map(i => i.toLong.toIntegral -> Random.nextDouble.toRealNN)
    val (scoresDF, f1, f2): (DataFrame, Feature[Integral], Feature[RealNN]) = TestFeatureBuilder(data)
    val percentile = f2.toPercentile()

    val model = percentile.originStage.asInstanceOf[Estimator[_]].fit(scoresDF)
    val scoresTransformed = model.asInstanceOf[Transformer].transform(scoresDF)

    val scoreCounts = scoresTransformed.groupBy(percentile.name)
      .agg(count(percentile.name))
      .withColumnRenamed(s"count(${percentile.name})", "count")

    val histBuckets = scoreCounts.select("count").map(_.getLong(0)).collect

    // compute the goodness of fit against a uniform distribution (results in line with the R command `chisq.test`)
    val goodnessOfFitTestResult = Statistics.chiSqTest(Vectors.dense(histBuckets.map(_.toDouble)))
    assert(goodnessOfFitTestResult.pValue > 0.5)
  }

  it should "return results in same order when sorted by probability or percentile" in {
    val data = (0 until 1000).map(i => i.toLong.toIntegral -> Random.nextDouble.toRealNN)
    val (scoresDF, f1, f2): (DataFrame, Feature[Integral], Feature[RealNN]) = TestFeatureBuilder(data)
    val percentile = f2.toPercentile()

    val model = percentile.originStage.asInstanceOf[Estimator[_]].fit(scoresDF)
    val scoresTransformed = model.asInstanceOf[Transformer].transform(scoresDF)

    val indicesByProb = scoresTransformed.orderBy(f2.name).collect(f1).deep
    val indicesByPerc = scoresTransformed.orderBy(percentile.name, f2.name).collect(f1).deep
    indicesByProb should equal (indicesByPerc)
  }
}
