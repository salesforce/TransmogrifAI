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
