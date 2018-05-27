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

package com.salesforce.op.stages.base.unary

import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.ml.param.ParamMap
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class UnaryTransformerTest extends FlatSpec with PassengerSparkFixtureTest {

  val scaleBy2 = new UnaryLambdaTransformer[Real, Real](operationName = "unary",
    transformFn = r => r.v.map(_ * 2.0).toReal
  )

  val toCat = new UnaryLambdaTransformer[Real, MultiPickList](operationName = "cat",
    transformFn = value => Set(value.v.getOrElse(0.0).toString).toMultiPickList
  )

  Spec[UnaryLambdaTransformer[_, _]] should "return single properly formed Feature" in {
    scaleBy2.setInput(weight)
    val feats = scaleBy2.getOutput()

    feats shouldBe new Feature[Real](
      name = scaleBy2.getOutputFeatureName,
      originStage = scaleBy2,
      isResponse = false,
      parents = Array(weight)
    )
  }

  it should "add column to DataFrame when transformed" in {
    scaleBy2.setInput(weight)
    val transformedData = scaleBy2.transform(passengersDataSet)
    val output = scaleBy2.getOutput()
    val answer = passengersArray.map(r => scaleBy2.transformFn(r.getFeatureType[Real](weight)))
    transformedData.columns.contains(scaleBy2.getOutputFeatureName) shouldBe true
    transformedData.collect(output) shouldBe answer
  }

  it should "work when returning a MultiPickList feature" in {
    toCat.setInput(weight)
    val transformedData = toCat.transform(passengersDataSet)
    val output = toCat.getOutput()
    val answer = passengersArray.map(r => toCat.transformFn(r.getFeatureType[Real](weight)))
    transformedData.columns.contains(toCat.getOutputFeatureName)  shouldBe true
    transformedData.collect(output) shouldBe answer
  }

  it should "copy successfully" in {
    val copy = scaleBy2.copy(new ParamMap())
    copy shouldBe a[UnaryTransformer[_, _]]
    copy.uid shouldBe scaleBy2.uid
  }

}
