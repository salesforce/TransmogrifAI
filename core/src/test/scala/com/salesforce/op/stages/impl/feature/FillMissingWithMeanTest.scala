/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.reflect.ClassTag


@RunWith(classOf[JUnitRunner])
class FillMissingWithMeanTest extends FlatSpec with TestSparkContext {
  val data = Seq[Real](Real(4.0), Real(2.0), Real.empty, Real(6.0))
  val dataNull = List.fill(7)(Real.empty)
  val binData = Seq[Binary](true.toBinary, false.toBinary, Binary.empty)

  lazy val (ds, f) = TestFeatureBuilder(data = data, f1name = "f")
  lazy val (dsi, fi) = TestFeatureBuilder(data = data.map(_.value.map(_.toLong).toIntegral), f1name = "fi")
  lazy val (dsNull, fNull) = TestFeatureBuilder(data = dataNull, f1name = "fNull")
  lazy val (dsb, fb) = TestFeatureBuilder(data = binData, f1name = "fb")

  Spec[FillMissingWithMean[_, _]] should "fill missing values with mean" in {
    assertUnaryEstimator[Real, RealNN](
      output = new FillMissingWithMean[Double, Real]().setInput(f).getOutput(),
      data = ds,
      expected = Array(4.0, 2.0, 4.0, 6.0).map(_.toRealNN)
    )
  }

  it should "fill missing values with mean from a shortcut on Real feature" in {
    assertUnaryEstimator[Real, RealNN](
      output = f.fillMissingWithMean(),
      data = ds,
      expected = Array(4.0, 2.0, 4.0, 6.0).map(_.toRealNN)
    )
  }

  it should "fill missing values with mean from a shortcut on Integral feature" in {
    assertUnaryEstimator[Real, RealNN](
      output = fi.fillMissingWithMean(),
      data = dsi,
      expected = Array(4.0, 2.0, 4.0, 6.0).map(_.toRealNN)
    )
  }

  it should "fill missing values with mean from a shortcut on Binary feature" in {
    assertUnaryEstimator[Real, RealNN](
      output = fb.fillMissingWithMean(),
      data = dsb,
      expected = Array(1.0, 0.0, 0.5).map(_.toRealNN)
    )
  }

  it should "fill a feature of only nulls with default value" in {
    val default = 3.14159
    assertUnaryEstimator[Real, RealNN](
      output = fNull.fillMissingWithMean(default = default),
      data = dsNull,
      expected = List.fill(dataNull.length)(default).map(_.toRealNN)
    )
  }

  // TODO move this assert to testkit
  private def assertUnaryEstimator[I <: FeatureType, O <: FeatureType : FeatureTypeSparkConverter : ClassTag]
  (
    output: FeatureLike[O], data: DataFrame, expected: Seq[O]
  ): Unit = {
    output.originStage shouldBe a[UnaryEstimator[_, _]]
    val model = output.originStage.asInstanceOf[UnaryEstimator[I, O]].fit(data)
    model shouldBe a[UnaryModel[_, _]]
    val transformed = model.asInstanceOf[UnaryModel[I, O]].transform(data)
    val results = transformed.collect(output)
    results should contain theSameElementsAs expected
  }
}
