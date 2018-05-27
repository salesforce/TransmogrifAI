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

package com.salesforce.op.stages.base.binary

import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.ml.param.ParamMap
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class BinaryTransformerTest extends FlatSpec with PassengerSparkFixtureTest {

  val bmi = new BinaryLambdaTransformer[Real, RealNN, Real](operationName = "bmi",
    transformFn = (i1, i2) => new Real(for { v1 <- i1.value; v2 <- i2.value } yield v1 / (v2 * v2))
  ).setInput(weight, height)

  Spec[BinaryLambdaTransformer[_, _, _]] should "return single properly formed Feature" in {
    val feats = bmi.getOutput()

    feats shouldBe new Feature[Real](
      name = bmi.getOutputFeatureName,
      originStage = bmi,
      isResponse = false,
      parents = Array(weight, height)
    )
  }

  it should "add column to DataFrame when transformed" in {
    val transformedData = bmi.transform(passengersDataSet)
    val columns = transformedData.columns
    assert(columns.contains(bmi.getOutputFeatureName))
    val output = bmi.getOutput()
    val answer = passengersArray.map(r =>
      bmi.transformFn(r.getFeatureType[Real](weight), r.getFeatureType[RealNN](height))
    )
    transformedData.collect(output) shouldBe answer
  }

  it should "copy successfully" in {
    val copy = bmi.copy(new ParamMap())
    copy.isInstanceOf[BinaryTransformer[_, _, _]] shouldBe true
    copy.uid shouldBe bmi.uid
  }
}
