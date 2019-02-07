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
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.apache.spark.SparkException
import org.joda.time.DateTime
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success}

@RunWith(classOf[JUnitRunner])
class DescalerTransformerTest extends OpTransformerSpec[Real, DescalerTransformer[Real, Real, Real]] {
  val (testData, f1) = TestFeatureBuilder(Seq(4.0, 1.0, 0.0).map(_.toReal))
  val scalerMetadata = ScalerMetadata(ScalingType.Linear, LinearScalerArgs(slope = 2.0, intercept = 3.0)).toMetadata()
  val colWithMetadata = testData.col(f1.name).as(f1.name, scalerMetadata)
  val inputData = testData.withColumn(f1.name, colWithMetadata)

  val transformer = new DescalerTransformer[Real, Real, Real]().setInput(f1, f1)
  val expectedResult: Seq[Real] = Seq(0.5, -1.0, -1.5).map(_.toReal)

  it should "error on missing scaler metadata" in {
    val (df, f) = TestFeatureBuilder(Seq(4.0, 1.0, 0.0).map(_.toReal))
    val error = intercept[SparkException](
      new DescalerTransformer[Real, Real, Real]().setInput(f, f).transform(df).collect()
    )
    error.getCause should not be null
    error.getCause shouldBe a[RuntimeException]
    error.getCause.getMessage shouldBe s"Failed to extract scaler metadata for input feature '${f1.name}'"
  }

  it should "descale and serialize log-scaling workflow" in {
    val logScaler = new ScalerTransformer[Real, Real](
      scalingType = ScalingType.Logarithmic,
      scalingArgs = EmptyArgs()
    ).setInput(f1)
    val scaledResponse = logScaler.getOutput()
    val metadata = logScaler.transform(inputData).schema(scaledResponse.name).metadata
    ScalerMetadata(metadata) match {
      case Failure(err) => fail(err)
      case Success(meta) =>
        meta shouldBe ScalerMetadata(ScalingType.Logarithmic, EmptyArgs())
    }

    val shifted = scaledResponse.map[Real](v => v.value.map(_ + 1).toReal, operationName = "shift")
    val descaledResponse = new DescalerTransformer[Real, Real, Real]().setInput(shifted, scaledResponse).getOutput()
    val workflow = new OpWorkflow().setResultFeatures(descaledResponse)
    val wfModel = workflow.setInputDataset(inputData).train()
    val modelLocation = tempDir + "logScalerDescalerTest" + DateTime.now().getMillis
    wfModel.save(modelLocation)

    val newModel = workflow.loadModel(modelLocation)
    val transformed = newModel.score()

    val actual = transformed.collect().map(_.getAs[Double](1))
    val expected = Array(4.0, 1.0, 0.0).map(_ * math.E)
    all(actual.zip(expected).map(x => math.abs(x._2 - x._1))) should be < 0.0001
  }
}
