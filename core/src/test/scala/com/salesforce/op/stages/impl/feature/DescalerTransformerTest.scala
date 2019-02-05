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

import com.salesforce.op.{OpWorkflow, OpWorkflowModelWriter}
import com.salesforce.op.features.{Feature, FeatureLike}
import com.salesforce.op.features.types.Real
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.utils.json.JsonUtils
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.joda.time.DateTime
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FlatSpec

@RunWith(classOf[JUnitRunner])
class  DescalerTransformerTest extends FlatSpec with TestSparkContext {
  val (testData, inA) = TestFeatureBuilder("inA", Seq[(Real)](Real(4.0), Real(1.0), Real(0.0)))
  val scalingType = "scalingType"
  val scalingArgs = "scalingArgs"

  Spec[DescalerTransformer[_, _, _]] should "Properly descale a linear scaling and serialize the workflow" in {
    val linearScaler = new ScalerTransformer[Real, Real](
      scalingType = ScalingType.Linear,
      scalingArgs = LinearScalerArgs(slope = 2.0, intercept = 3.0)
    ).setInput(inA.asInstanceOf[Feature[Real]])
    val scaledResponse = linearScaler.getOutput().asInstanceOf[Feature[Real]]
    val shifter = new UnaryLambdaTransformer[Real, Real](
      operationName = "shift",
      transformFn = v => Real(v.value.get + 1)
    ).setInput(scaledResponse)
    val metadata = linearScaler.transform(testData).schema(scaledResponse.name).metadata
    metadata.getString(scalingType) shouldEqual ScalingType.Linear.entryName
    val args = JsonUtils.fromString[LinearScalerArgs](metadata.getString(scalingArgs)).get
    args.slope shouldEqual 2.0
    args.intercept shouldEqual 3.0
    val shifted = shifter.getOutput()
    val deScaler = new DescalerTransformer[Real, Real, Real]().setInput(shifted, scaledResponse)
    val descaledResponse = deScaler.getOutput()
    val wfModel = new OpWorkflow().setResultFeatures(descaledResponse).setInputDataset(testData).train()
    wfModel.save(tempDir + "linearScalerDescalerTest" + DateTime.now().getMillis)
    val data = wfModel.score()
    val actual = data.collect().map(_.getAs[Double](1))
    actual shouldEqual Array(4.0, 1.0, 0.0).map(_ + 0.5)
  }

  it should "Properly descale a log scaling and serialize the workflow" in {
    val logScaler = new ScalerTransformer[Real, Real](
      scalingType = ScalingType.Logarithmic,
      scalingArgs = EmptyArgs()
    ).setInput(inA.asInstanceOf[Feature[Real]])
    val scaledResponse = logScaler.getOutput().asInstanceOf[Feature[Real]]
    val shifter = new UnaryLambdaTransformer[Real, Real](
      operationName = "shift",
      transformFn = v => Real(v.value.get + 1)
    ).setInput(scaledResponse)
    val metadata = logScaler.transform(testData).schema(scaledResponse.name).metadata
    metadata.getString(scalingType) shouldEqual ScalingType.Logarithmic.entryName
    val args = JsonUtils.fromString[EmptyArgs](metadata.getString(scalingArgs)).get
    val shifted = shifter.getOutput()
    val deScaler = new DescalerTransformer[Real, Real, Real]().setInput(shifted, scaledResponse)
    val descaledResponse = deScaler.getOutput()
    val wfModel = new OpWorkflow().setResultFeatures(descaledResponse).setInputDataset(testData).train()
    wfModel.save(tempDir + "linearScalerDescalerTest" + DateTime.now().getMillis)
    val data = wfModel.score()
    val actual = data.collect().map(_.getAs[Double](1))
    val expected = Array(4.0, 1.0, 0.0).map(_ * math.E)
    all(actual.zip(expected).map(x => math.abs(x._2 - x._1))) should be < 0.0001
  }
}

