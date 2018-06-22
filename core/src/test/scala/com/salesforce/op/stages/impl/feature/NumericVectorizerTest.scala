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
import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.test.{FeatureTestBase, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.testkit.{RandomIntegral, RandomReal}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class NumericVectorizerTest extends FlatSpec with FeatureTestBase {

  val ageData: Seq[Real] = RandomReal.uniform[Real](maxValue = 80.0).limit(100)
  val heightData: Seq[Real] = RandomReal.normal[Real](mean = 65.0, sigma = 8).limit(100)
  val countData: Seq[Integral] = RandomIntegral.integrals(0, 10).limit(100)
  val labelTransformer = new UnaryLambdaTransformer[Real, RealNN](operationName = "labelFunc",
    transformFn = {
      case SomeValue(Some(x)) if x > 30.0 => 1.toRealNN
      case _ => 0.0.toRealNN
    }
  )

  Spec[RichRealFeature[_]] should "vectorize a small sample of real values" in {
    val inputData = Seq(-4, -3, -2, -1, 1, 2, 3, 4).map(_.toReal)
    val labelData = Seq(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0).map(_.toRealNN)
    val generatedData = inputData.zip(labelData)
    val (ds, input, label) = TestFeatureBuilder("input", "label", generatedData)
    val autoBucketFeature = Seq(input).transmogrify(label = Some(label.copy(isResponse = true)))
    val vectorized = new OpWorkflow().setResultFeatures(autoBucketFeature).transform(ds)
    // value col, null indicator col, bucket 0 indicator, bucket 1 indicator
    val expected = Array(
      Array(-4.0, 0.0, 1.0, 0.0),
      Array(-3.0, 0.0, 1.0, 0.0),
      Array(-2.0, 0.0, 1.0, 0.0),
      Array(-1.0, 0.0, 1.0, 0.0),
      Array(1.0, 0.0, 0.0, 1.0),
      Array(2.0, 0.0, 0.0, 1.0),
      Array(3.0, 0.0, 0.0, 1.0),
      Array(4.0, 0.0, 0.0, 1.0)
    ).map(Vectors.dense(_).toOPVector)
    vectorized.collect(autoBucketFeature) should contain theSameElementsAs expected
  }
  it should "vectorize single real feature with a label" in {
    val (ds, age) = TestFeatureBuilder("age", ageData)
    val labelData = age.transformWith(labelTransformer).asInstanceOf[Feature[RealNN]].copy(isResponse = true)
    val autoBucketFeature = Seq(age).transmogrify(label = Some(labelData))
    val manualBucketFeature = Seq(
      age.vectorize(fillValue = 0, fillWithMean = true, trackNulls = true),
      age.autoBucketize(labelData, trackNulls = false)
    ).combine()
    val vectorized = new OpWorkflow().setResultFeatures(autoBucketFeature, manualBucketFeature).transform(ds)

    for {(autoAge, manualAge) <- vectorized.collect(autoBucketFeature, manualBucketFeature)} {
      autoAge.v.toArray should contain theSameElementsAs manualAge.v.toArray
    }
  }
  it should "vectorize multiple real features with a label" in {
    val generatedData: Seq[(Real, Real)] = ageData.zip(heightData)
    val (ds, age, height) = TestFeatureBuilder("age", "height", generatedData)
    val labelData = age.transformWith(labelTransformer).asInstanceOf[Feature[RealNN]].copy(isResponse = true)
    val autoBucketFeature = Seq(age, height).transmogrify(label = Some(labelData))
    val manualBucketFeature = Seq(
      age, age.autoBucketize(labelData, trackNulls = false),
      height, height.autoBucketize(labelData, trackNulls = false)
    ).transmogrify()
    val vectorized = new OpWorkflow().setResultFeatures(autoBucketFeature, manualBucketFeature).transform(ds)

    for {(autoAge, manualAge) <- vectorized.collect(autoBucketFeature, manualBucketFeature)} {
      autoAge.v.toArray should contain theSameElementsAs manualAge.v.toArray
    }
  }
  Spec[RichIntegralFeature[_]] should "vectorize single integral feature with a label" in {
    val (ds, count) = TestFeatureBuilder("count", countData)
    val labelTransformer = new UnaryLambdaTransformer[Integral, RealNN](operationName = "labelFunc",
      transformFn = {
        case SomeValue(Some(x)) if x > 5 => 1.0.toRealNN
        case _ => 0.0.toRealNN
      }
    )
    val labelData = labelTransformer.setInput(count).getOutput().asInstanceOf[Feature[RealNN]].copy(isResponse = true)
    val autoBucketFeature = Seq(count).transmogrify(label = Some(labelData))
    val manualBucketFeature = Seq(count, count.autoBucketize(labelData, trackNulls = false)).transmogrify()
    val vectorized = new OpWorkflow().setResultFeatures(autoBucketFeature, manualBucketFeature).transform(ds)

    for {(autoAge, manualAge) <- vectorized.collect(autoBucketFeature, manualBucketFeature)} {
      autoAge.v.toArray should contain theSameElementsAs manualAge.v.toArray
    }
  }
}
