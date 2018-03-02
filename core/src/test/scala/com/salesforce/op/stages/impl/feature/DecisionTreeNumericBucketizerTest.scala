/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomBinary, RandomReal}
import com.salesforce.op.utils.numeric.Number
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DecisionTreeNumericBucketizerTest extends FlatSpec with TestSparkContext {

  trait NormalData {
    val numericData: Seq[Real] = RandomReal.normal[Real]().withProbabilityOfEmpty(0.2).limit(1000)
    val labelData: Seq[RealNN] = RandomBinary(probabilityOfSuccess = 0.4).limit(1000).map(_.toDouble.toRealNN)
    val (ds, numeric, label) = TestFeatureBuilder[Real, RealNN](numericData zip labelData)
    val expectedSplits = Array.empty[Double]
  }

  trait EmptyData {
    val (_, numeric, label) = TestFeatureBuilder[Real, RealNN](Seq[Real]() zip Seq[RealNN]())
    val nulls = Seq[(java.lang.Double, java.lang.Double)]((1.0, null), (null, null))
    import spark.implicits._
    val ds = spark.createDataset(nulls).toDF(label.name, numeric.name)
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

  Spec[DecisionTreeNumericBucketizer[_, _]] should "produce output that is never a response, " +
    "except the case where both inputs are" in new NormalData {
    Seq(
      label.copy(isResponse = false) -> numeric.copy(isResponse = false),
      label.copy(isResponse = true) -> numeric.copy(isResponse = false),
      label.copy(isResponse = false) -> numeric.copy(isResponse = true)
    ).foreach(inputs =>
      new DecisionTreeNumericBucketizer[Double, Real]().setInput(inputs).getOutput().isResponse shouldBe false
    )
    Seq(
      label.copy(isResponse = true) -> numeric.copy(isResponse = true)
    ).foreach(inputs =>
      new DecisionTreeNumericBucketizer[Double, Real]().setInput(inputs).getOutput().isResponse shouldBe true
    )
  }

  it should "should not find any splits on random data" in new NormalData {
    val bucketizer = new DecisionTreeNumericBucketizer[Double, Real]().setInput(label, numeric).setTrackNulls(false)
    assertBucketizer(
      bucketizer, data = ds, shouldSplit = false, trackNulls = false, trackInvalid = false,
      expectedSplits = Array.empty, expectedTolerance = 0.0
    )
  }

  it should "work correctly on empty data" in new EmptyData {
    val bucketizer = new DecisionTreeNumericBucketizer[Double, Real]().setInput(label, numeric).setTrackNulls(false)
    assertBucketizer(
      bucketizer, data = ds, shouldSplit = false, trackNulls = false, trackInvalid = false,
      expectedSplits = Array.empty, expectedTolerance = 0.0
    )
  }

  it should "work as a shortcut" in new NormalData {
    val out = numeric.autoBucketize(label, trackNulls = false)
    out.originStage shouldBe a[DecisionTreeNumericBucketizer[_, _]]
    assertBucketizer(
      bucketizer = out.originStage.asInstanceOf[DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]]],
      data = ds, shouldSplit = false, trackNulls = false, trackInvalid = false,
      expectedSplits = Array.empty, expectedTolerance = 0.0
    )
  }

  it should "correctly bucketize when labels are specified" in new UniformData {
    val out = currency.autoBucketize(label = label, trackNulls = false, minInfoGain = 0.1)
    assertBucketizer(
      bucketizer = out.originStage.asInstanceOf[DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]]],
      data = ds, shouldSplit = true, trackNulls = false, trackInvalid = false, expectedSplits = expectedSplits,
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

    val out = rawER.autoBucketize(label.copy(isResponse = true), trackNulls = true, trackInvalid = true)
    assertBucketizer(
      bucketizer = out.originStage.asInstanceOf[DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]]],
      data = ds, shouldSplit = true, trackNulls = true, trackInvalid = true,
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
    trackInvalid: Boolean,
    expectedSplits: Array[Double],
    expectedTolerance: Double
  ): Unit = {
    val out = bucketizer.getOutput()
    val fitted = bucketizer.fit(data)
    fitted shouldBe a[DecisionTreeNumericBucketizerModel[_, _]]
    val model = fitted.asInstanceOf[DecisionTreeNumericBucketizerModel[_, _]]
    model.shouldSplit shouldBe shouldSplit
    model.trackNulls shouldBe trackNulls
    model.trackInvalid shouldBe trackInvalid

    val splits = model.splits
    assertSplits(splits = splits, expectedSplits = expectedSplits, expectedTolerance)

    val meta = model.getMetadata().getSummaryMetadata()
    meta.getBoolean(DecisionTreeNumericBucketizer.ShouldSplitKey) shouldBe shouldSplit
    val metaSplits = meta.getDoubleArray(DecisionTreeNumericBucketizer.SplitsKey)
    splits shouldBe metaSplits
    assertSplits(splits = metaSplits, expectedSplits = expectedSplits, expectedTolerance)

    val res = model.transform(data).collect(out)
    if (shouldSplit) {
      val expectedSize = splits.length - 1 + (if (trackNulls) 1 else 0) + (if (trackInvalid) 1 else 0)
      for {v <- res} v.value.size shouldBe expectedSize

      val columnMeta: Array[OpVectorColumnMetadata] =
        model.getMetadata().getMetadataArray(OpVectorMetadata.ColumnsKey)
          .flatMap(OpVectorColumnMetadata.fromMetadata)

      columnMeta.length shouldBe expectedSize
    }
    else {
      res.foreach(_.value.size shouldBe 0)
      model.getMetadata().getMetadataArray(OpVectorMetadata.ColumnsKey) shouldBe Nil
    }
  }

  private def assertSplits(splits: Array[Double], expectedSplits: Array[Double], expectedTolerance: Double): Unit = {
    withClue(
      s"Bucketizer splits: ${splits.mkString("[", ",", "]")}, expected: ${expectedSplits.mkString("[", ",", "]")}"
    ) {
      // account for an potential extra split
      (splits.length == expectedSplits.length || splits.length == expectedSplits.length + 1) shouldBe true
      // relative difference check per value
      val diff = splits.zip(expectedSplits).map { case (s, e) => math.abs((s - e) / math.max(s, e)) }
      diff.filter(Number.isValid).foreach(_ should be <= expectedTolerance)
    }
  }

}
