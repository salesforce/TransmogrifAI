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

package com.salesforce.op.stages

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.stages.impl.feature.{DescalerTransformer, LinearScalerArgs, ScalerMetadata, ScalingType, StandardMinEstimator}
import com.salesforce.op.test.TestFeatureBuilder
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success}


@RunWith(classOf[JUnitRunner])
class OpMinMaxEstimatorReaderWriterTest extends OpPipelineStageReaderWriterTest {
  private val minMax = new MinMaxNormEstimator().setInput(weight).setMetadata(meta)

  lazy val stage = minMax.fit(passengersDataSet)

  val expected =
    Array(1.0.toReal, 0.0.toReal, Real.empty, 0.10476190476190476.toReal, 0.2761904761904762.toReal, 0.0.toReal)

  val inputValues: Seq[Double] = Seq(10, 100, 1000)

  val (inputData, testF) = TestFeatureBuilder(inputValues.map(_.toReal))

  it should "descale and work in min-max normalized workflow" in {
    val featureNormalizer = new MinMaxNormEstimator().setInput(testF)
    val normedOutput = featureNormalizer.getOutput()
    val metadata = featureNormalizer.fit(inputData).getMetadata()

    val expectedMin = inputValues.min
    val expectedMax = inputValues.max
    val expectedSlope = 1 / (expectedMax - expectedMin)
    val expectedIntercept = - expectedMin / (expectedMax - expectedMin)
    ScalerMetadata(metadata) match {
      case Failure(err) => fail(err)
      case Success(meta) =>
        meta shouldBe a[ScalerMetadata]
        meta.scalingType shouldBe ScalingType.Linear
        meta.scalingArgs shouldBe a[LinearScalerArgs]
        math.abs((meta.scalingArgs.asInstanceOf[LinearScalerArgs].slope - expectedSlope)
          / expectedSlope) should be < 0.001
        math.abs((meta.scalingArgs.asInstanceOf[LinearScalerArgs].intercept - expectedIntercept)
          / expectedIntercept) should be < 0.001
    }

    val descaledResponse = new DescalerTransformer[Real, Real, Real]()
      .setInput(normedOutput, normedOutput).getOutput()
    val workflow = new OpWorkflow().setResultFeatures(descaledResponse)
    val wfModel = workflow.setInputDataset(inputData).train()
    val transformed = wfModel.score()

    val actual = transformed.collect().map(_.getAs[Double](1))
    val expected : Seq[Double] = inputValues
    all(actual.zip(expected).map(x => math.abs(x._2 - x._1))) should be < 0.0001
  }
}


class MinMaxNormEstimator(uid: String = UID[MinMaxNormEstimator])
  extends UnaryEstimator[Real, Real](operationName = "minMaxNorm", uid = uid) {

  def fitFn(dataset: Dataset[Real#Value]): UnaryModel[Real, Real] = {
    val grouped = dataset.groupBy()
    val maxVal = grouped.max().first().getDouble(0)
    val minVal = grouped.min().first().getDouble(0)

    val scalingArgs = LinearScalerArgs(1 / (maxVal - minVal), - minVal / (maxVal - minVal))
    val meta = ScalerMetadata(ScalingType.Linear, scalingArgs).toMetadata()
    setMetadata(meta)

    new MinMaxNormEstimatorModel(
      min = minVal,
      max = maxVal,
      seq = Seq(minVal, maxVal),
      map = Map("a" -> Map("b" -> 1.0, "c" -> 2.0), "d" -> Map.empty),
      operationName = operationName,
      uid = uid
    )
  }
}

final class MinMaxNormEstimatorModel private[op]
(
  val min: Double,
  val max: Double,
  val seq: Seq[Double],
  val map: Map[String, Map[String, Double]],
  operationName: String, uid: String
) extends UnaryModel[Real, Real](operationName = operationName, uid = uid) {
  def transformFn: Real => Real = r => r.v.map(v => (v - min) / (max - min)).toReal
}
