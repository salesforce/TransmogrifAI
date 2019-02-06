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

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types.{Prediction, Real}
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.utils.json.JsonUtils
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.joda.time.DateTime
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class  DescalerTransformerTest extends OpTransformerSpec[Real, DescalerTransformer[Real, Real, Real]] {
  val scalingType = "scalingType"
  val scalingArgs = "scalingArgs"
  val (testData, f1) = TestFeatureBuilder("f1", Seq[(Real)](Real(4.0), Real(1.0), Real(0.0)))
  val scalerMetadata = ScalerMetadata(scalingType = ScalingType.Linear,
    scalingArgs = LinearScalerArgs(slope = 2.0, intercept = 3.0)
  ).toMetadata()
  val colWithMetadata = testData.col(f1.name).as(f1.name, scalerMetadata)
  val inputData = testData.withColumn(f1.name, colWithMetadata)
  val expectedResult: Seq[Real] = Seq(0.5, -1.0, -1.5).map(Real(_))
  val transformer = new DescalerTransformer[Real, Real, Real]().setInput(f1, f1)

  it should "Properly descale and serialize log-scaling workflow" in {
    val logScaler = new ScalerTransformer[Real, Real](
      scalingType = ScalingType.Logarithmic,
      scalingArgs = EmptyArgs()
    ).setInput(f1.asInstanceOf[Feature[Real]])
    val scaledResponse = logScaler.getOutput().asInstanceOf[Feature[Real]]
    val metadata = logScaler.transform(inputData).schema(scaledResponse.name).metadata
    metadata.getString(scalingType) shouldEqual ScalingType.Logarithmic.entryName
    val args = JsonUtils.fromString[EmptyArgs](metadata.getString(scalingArgs)).get
    args shouldEqual EmptyArgs()
    val shifted = new UnaryLambdaTransformer[Real, Real](
      operationName = "shift",
      transformFn = v => Real(v.value.get + 1)
    ).setInput(scaledResponse).getOutput()
    val descaledResponse = new DescalerTransformer[Real, Real, Real]().setInput(shifted, scaledResponse).getOutput()
    val wfModel = new OpWorkflow().setResultFeatures(descaledResponse).setInputDataset(inputData).train()
    wfModel.save(tempDir + "logScalerDescalerTest" + DateTime.now().getMillis)
    val transformed = wfModel.score()
    val actual = transformed.collect().map(_.getAs[Double](1))
    val expected = Array(4.0, 1.0, 0.0).map(_ * math.E)
    all(actual.zip(expected).map(x => math.abs(x._2 - x._1))) should be < 0.0001
  }
}

