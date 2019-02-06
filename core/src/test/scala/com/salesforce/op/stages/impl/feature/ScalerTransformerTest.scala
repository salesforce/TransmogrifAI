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

import com.salesforce.op.features.{Feature, FeatureLike}
import com.salesforce.op.features.types.Real
import com.salesforce.op.utils.json.JsonUtils
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class  ScalerTransformerTest extends OpTransformerSpec[Real, ScalerTransformer[Real, Real]] {
  val (inputData, f1) = TestFeatureBuilder("f1", Seq[(Real)](Real(4.0), Real(1.0), Real(0.0)))
  val scalingType = "scalingType"
  val scalingArgs = "scalingArgs"
  val transformer = new ScalerTransformer[Real, Real](
    scalingType = ScalingType.Linear,
    scalingArgs = LinearScalerArgs(slope = 2.0, intercept = 1.0)
  ).setInput(f1.asInstanceOf[Feature[Real]])

  override val expectedResult: Seq[Real] = Seq(9.0, 3.0, 1.0).map(Real(_))

  it should "Properly linearly scale numeric fields" in {
    val vector: FeatureLike[Real] = transformer.getOutput()
    val transformed = transformer.transform(inputData)
    val actual = transformed.collect()
    val metadata = transformed.schema(vector.name).metadata
    metadata.getString(scalingType) shouldEqual ScalingType.Linear.entryName
    val args: LinearScalerArgs = JsonUtils.fromString[LinearScalerArgs](metadata.getString(scalingArgs)).get
    args.intercept shouldEqual 1.0
    args.slope shouldEqual 2.0
  }

  it should "Properly log scale numeric fields and serialize the transformer" in {
    val logScaler = new ScalerTransformer[Real, Real](
      scalingType = ScalingType.Logarithmic,
      scalingArgs = EmptyArgs()
    ).setInput(f1.asInstanceOf[Feature[Real]])
    val vector: FeatureLike[Real] = logScaler.getOutput()
    val transformed = logScaler.transform(inputData)
    val actual = transformed.collect()
    actual.map(_.getAs[Double](1)) shouldEqual Array(4.0, 1.0, 0.0).map(math.log(_))
    val metadata = transformed.schema(vector.name).metadata
    metadata.getString(scalingType) shouldEqual ScalingType.Logarithmic.entryName
    val args: EmptyArgs = JsonUtils.fromString[EmptyArgs](metadata.getString(scalingArgs)).get
  }
}
