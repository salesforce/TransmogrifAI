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

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryLambdaTransformer
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.tuples.RichTuple._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class AliasTransformerTest extends OpTransformerSpec[RealNN, AliasTransformer[RealNN]] {
  val sample = Seq((RealNN(1.0), RealNN(2.0)), (RealNN(4.0), RealNN(4.0)))
  val (inputData, f1, f2) = TestFeatureBuilder(sample)
  val transformer = new AliasTransformer(name = "feature").setInput(f1)
  val expectedResult: Seq[RealNN] = sample.map(_._1)

  it should "have a shortcut that changes feature name on a raw feature" in {
    val feature = f1.alias
    feature.name shouldBe "feature"
    feature.originStage shouldBe a[AliasTransformer[_]]
    val origin = feature.originStage.asInstanceOf[AliasTransformer[RealNN]]
    val transformed = origin.transform(inputData)
    transformed.collect(feature) shouldEqual expectedResult
  }
  it should "have a shortcut that changes feature name on a derived feature" in {
    val feature = (f1 / f2).alias
    feature.name shouldBe "feature"
    feature.originStage shouldBe a[BinaryLambdaTransformer[_, _, _]]
    val origin = feature.originStage.asInstanceOf[BinaryLambdaTransformer[_, _, _]]
    val transformed = origin.transform(inputData)
    transformed.columns should contain (feature.name)
    transformed.collect(feature) shouldEqual sample.map { case (v1, v2) => (v1.v -> v2.v).map(_ / _).toRealNN(0.0) }
  }
  it should "have a shortcut that changes feature name on a derived wrapped feature" in {
    val feature = f1.toIsotonicCalibrated(label = f2).alias
    feature.name shouldBe "feature"
    feature.originStage shouldBe a[AliasTransformer[_]]
  }
}
