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
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.SparkException
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success}

@RunWith(classOf[JUnitRunner])
class PredictionDescalerTransformerTest extends OpTransformerSpec[Real, PredictionDescaler[Real, Real]] {
  val predictionData = Seq(-1.0, 0.0, 1.0, 2.0).map(Prediction(_))
  val featureData = Seq(0.0, 1.0, 2.0, 3.0).map(_.toReal)
  val (testData, p, f1) = TestFeatureBuilder[Prediction, Real](predictionData zip featureData)

  val scalerMetadata = ScalerMetadata(ScalingType.Linear, LinearScalerArgs(slope = 4.0, intercept = 1.0)).toMetadata()
  val colWithMetadata = testData.col(f1.name).as(f1.name, scalerMetadata)
  val inputData = testData.withColumn(f1.name, colWithMetadata)

  val transformer = new PredictionDescaler[Real, Real]().setInput(p, f1)

  val expectedResult: Seq[Real] = Seq(-0.5, -0.25, 0.0, 0.25).map(_.toReal)

  it should "error on missing scaler metadata" in {
    val (df, p, f1) = TestFeatureBuilder(Seq(4.0, 1.0, 0.0).map(Prediction(_)) zip Seq(0.0, 0.0, 0.0).map(Real(_)))
    val error = intercept[SparkException](
      new PredictionDescaler[Real, Real]().setInput(p, f1).transform(df).collect()
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
    val shifted = scaledResponse.map[Prediction](v => Prediction(v.value.getOrElse(Double.NaN) + 1),
      operationName = "shift")
    val descaledPrediction = new PredictionDescaler[Real, Real]().setInput(shifted, scaledResponse).getOutput()
    val workflow = new OpWorkflow().setResultFeatures(descaledPrediction)
    val wfModel = workflow.setInputDataset(inputData).train()
    val transformed = wfModel.score()
    val actual = transformed.collect().map(_.getAs[Double](1))
    val expected = Array(0.0, 1.0, 2.0, 3.0).map(_ * math.E)
    all(actual.zip(expected).map(x => math.abs(x._2 - x._1))) should be < 0.0001
  }

  it should "descale and serialize linear-scaling workflow" in {
    val scalingArgs = LinearScalerArgs(slope = 2.0, intercept = 0.0)
    val linearScaler = new ScalerTransformer[Real, Real](
      scalingType = ScalingType.Linear,
      scalingArgs = scalingArgs
    ).setInput(f1)
    val scaledResponse = linearScaler.getOutput()
    val metadata = linearScaler.transform(inputData).schema(scaledResponse.name).metadata
    ScalerMetadata(metadata) match {
      case Failure(err) => fail(err)
      case Success(meta) =>
        meta shouldBe ScalerMetadata(ScalingType.Linear, scalingArgs)
    }
    val shifted = scaledResponse.map[Prediction](v => Prediction(v.value.getOrElse(Double.NaN) + 1),
      operationName = "shift")
    val descaledPrediction = new PredictionDescaler[Real, Real]().setInput(shifted, scaledResponse).getOutput()
    val workflow = new OpWorkflow().setResultFeatures(descaledPrediction)
    val wfModel = workflow.setInputDataset(inputData).train()
    val transformed = wfModel.score()
    val actual = transformed.collect().map(_.getAs[Double](1))
    val expected = Array(0.5, 1.5, 2.5, 3.5)
    actual shouldBe expected
  }

  it should "work with its shortcut" in {
    val descaled = p.descale[Real, Real](f1)
    val transformed = descaled.originStage.asInstanceOf[PredictionDescaler[Real, Real]].transform(inputData)
    val actual = transformed.collect(descaled)
    actual shouldEqual expectedResult.toArray
  }

}
