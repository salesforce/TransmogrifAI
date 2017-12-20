/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomBinary, RandomReal}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.dsl.Number

@RunWith(classOf[JUnitRunner])
class DecisionTreeNumericBucketizerTest extends FlatSpec with TestSparkContext {

  trait NormalData {
    val numericData: Seq[Real] = RandomReal.normal[Real]().withProbabilityOfEmpty(0.2).limit(1000)
    val labelData: Seq[RealNN] = RandomBinary(probabilityOfSuccess = 0.4).limit(1000).map(_.toDouble.toRealNN)
    val (ds, numeric, label) = TestFeatureBuilder[Real, RealNN](numericData zip labelData)
    val expectedSplits = Array.empty[Double]
  }

  trait UniformData {
    val (min, max) = (0.0, 100.0)
    val currencyData: Seq[Currency] = RandomReal.uniform[Currency](minValue = min, maxValue = max).limit(1000)
    val labelData = currencyData.map(c => {
      c.value.map {
        case v if v < 15 => 0.0
        case v if v < 26 => 1.0
        case v if v < 91 => 2.0
        case _ => 3.0
      }.toRealNN
    })
    val (ds, currency, label) = TestFeatureBuilder("currency", "label", currencyData zip labelData)
    val expectedSplits = Array(Double.NegativeInfinity, 15, 26, 91, Double.PositiveInfinity)
  }

  Spec[DecisionTreeNumericBucketizer[_, _]] should "should not find any splits on random data" in new NormalData {
    val bucketizer = new DecisionTreeNumericBucketizer[Double, Real]().setInput(label, numeric).setTrackNulls(false)
    assertBucketizer(
      bucketizer, data = ds, shouldSplit = false, trackNulls = false, expectedSplits = Array.empty,
      expectedTolerance = 0.0
    )
  }

  it should "work as a shortcut" in new NormalData {
    val out = numeric.autoBucketize(label, trackNulls = false)
    out.originStage shouldBe a[DecisionTreeNumericBucketizer[_, _]]
    assertBucketizer(
      bucketizer = out.originStage.asInstanceOf[DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]]],
      data = ds, shouldSplit = false, trackNulls = false, expectedSplits = Array.empty,
      expectedTolerance = 0.0
    )
  }

  it should "correctly bucketize when labels are specified" in new UniformData {
    val out = currency.autoBucketize(label, trackNulls = false)
    assertBucketizer(
      bucketizer = out.originStage.asInstanceOf[DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]]],
      data = ds, shouldSplit = true, trackNulls = false, expectedSplits = expectedSplits,
      expectedTolerance = 0.15
    )
  }

  it should "correctly bucketize over data with label leakage" in {
    val binaryData: Seq[Binary] = RandomBinary(probabilityOfSuccess = 0.5).withProbabilityOfEmpty(0.3).limit(1000)
    val currencyData: Seq[Currency] = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).limit(1000)
    val expectedRevenueData: Seq[Currency] = currencyData.zip(binaryData).map { case (cur, bi) =>
      bi.value match {
        case Some(v) => ((if (v) 1.0 else 0.0) * cur.value.get).toCurrency
        case None => Currency.empty
      }
    }
    val labelData = binaryData.map(_.toDouble.getOrElse(0.0).toRealNN)
    val rawData: Seq[(Binary, Currency, Currency, RealNN)] =
      binaryData.zip(currencyData).zip(expectedRevenueData).zip(labelData).map {
        case (((bi, cu), er), l) => (bi, cu, er, l)
      }

    val (ds, rawBinary, rawCurrency, rawER, label) =
      TestFeatureBuilder("binary", "currency", "expectedRevenue", "label", rawData)

    val out = rawER.autoBucketize(label.copy(isResponse = true), trackNulls = true)
    assertBucketizer(
      bucketizer = out.originStage.asInstanceOf[DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]]],
      data = ds, shouldSplit = true, trackNulls = true,
      expectedSplits = Array(Double.NegativeInfinity, 0.0, Double.PositiveInfinity),
      expectedTolerance = 0.15
    )
  }


  private def assertBucketizer
  (
    bucketizer: DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]],
    data: DataFrame,
    shouldSplit: Boolean,
    trackNulls: Boolean,
    expectedSplits: Array[Double],
    expectedTolerance: Double
  ) = {
    val out = bucketizer.getOutput()
    val fitted = bucketizer.fit(data)
    fitted shouldBe a[SmartNumericBucketizerModel[_, _]]
    val model = fitted.asInstanceOf[SmartNumericBucketizerModel[_, _]]
    model.shouldSplit shouldBe shouldSplit
    model.trackNulls shouldBe trackNulls

    val splits = model.splits
    withClue(
      s"Bucketizer splits: ${splits.mkString("[", ",", "]")}, " +
      s"expected: ${expectedSplits.mkString("[", ",", "]")}"
    ) {
      // account for an potential extra split
      (splits.length == expectedSplits.length || splits.length == expectedSplits.length + 1) shouldBe true
      // relative difference check per value
      val diff = splits.zip(expectedSplits).map { case (s, e) => math.abs((s - e) / math.max(s, e)) }
      diff.filter(Number.isValid).foreach(_ should be <= expectedTolerance)
    }

    val res = model.transform(data).collect(out)
    if (shouldSplit) res.foreach(_.value.size shouldBe splits.length - 1 + (if (trackNulls) 1 else 0))
    else res.foreach(_.value.size shouldBe 0)
  }

}
