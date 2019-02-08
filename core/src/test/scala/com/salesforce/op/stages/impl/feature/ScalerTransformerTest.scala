/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success}

@RunWith(classOf[JUnitRunner])
class  ScalerTransformerTest extends OpTransformerSpec[Real, ScalerTransformer[Real, Real]] {
  val (inputData, f1) = TestFeatureBuilder(Seq(4.0, 1.0, 0.0).toReal)

  val transformer = new ScalerTransformer[Real, Real](
    scalingType = ScalingType.Linear,
    scalingArgs = LinearScalerArgs(slope = 2.0, intercept = 1.0)
  ).setInput(f1)

  val expectedResult: Seq[Real] = Seq(9.0, 3.0, 1.0).map(_.toReal)

  it should "linearly scale numeric fields and produce correct metadata" in {
    val scaled: FeatureLike[Real] = transformer.getOutput()
    val transformed = transformer.transform(inputData)

    val actual = transformed.collect(scaled)
    actual shouldEqual expectedResult

    ScalerMetadata(transformed.schema(scaled.name).metadata) match {
      case Failure(err) => fail(err)
      case Success(meta) =>
        meta shouldBe ScalerMetadata(ScalingType.Linear, LinearScalerArgs(slope = 2.0, intercept = 1.0))
    }
  }

  it should "log scale numeric fields and produce correct metadata" in {
    val logScaler = new ScalerTransformer[Real, Real](
      scalingType = ScalingType.Logarithmic, scalingArgs = EmptyArgs()
    ).setInput(f1)

    val scaled: FeatureLike[Real] = logScaler.getOutput()
    val transformed = logScaler.transform(inputData)

    val actual = transformed.collect(scaled)
    actual shouldEqual Array(4.0, 1.0, 0.0).map(math.log).map(_.toReal)

    ScalerMetadata(transformed.schema(scaled.name).metadata) match {
      case Failure(err) => fail(err)
      case Success(meta) =>
        meta shouldBe ScalerMetadata(ScalingType.Logarithmic, EmptyArgs())
    }
  }

  it should "work with its shortcut" in {
    val scaled = f1.scale(scalingType = ScalingType.Linear,
      scalingArgs= LinearScalerArgs(slope = 10.0, intercept = 0.5)
    )
    val transformed = scaled.originStage.asInstanceOf[ScalerTransformer[RealNN, RealNN]].transform(inputData)
    val actual = transformed.collect(scaled)
    actual shouldEqual Array(40.5, 10.5, 0.5).map(_.toRealNN)
  }
}
