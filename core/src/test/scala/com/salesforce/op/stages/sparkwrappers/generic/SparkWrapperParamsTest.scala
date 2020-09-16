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

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.test.PassengerFeaturesTest.HeightToRealNNExtract
import com.salesforce.op.test.{Passenger, TestCommon, TestSparkContext}
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterEach, FlatSpec}

@RunWith(classOf[JUnitRunner])
class SparkWrapperParamsTest extends FlatSpec with BeforeAndAfterEach with TestSparkContext {

  private def estimator(sparkMlStageIn: Option[StandardScaler] = None) = {
    new SwUnaryEstimator[Real, Real, StandardScalerModel, StandardScaler](
      inputParamName = "in", outputParamName = "out",
      operationName = "test-op", sparkMlStageIn = sparkMlStageIn
    )
  }
  private val feature = FeatureBuilder.Real[Passenger].extract(new HeightToRealNNExtract).asPredictor

  Spec[SparkWrapperParams[_]] should "have proper default values for path and stage" in {
    val stage = estimator()
    stage.getStageSavePath() shouldBe None
    stage.getSparkMlStage() shouldBe None
  }
  it should "when setting path, it should also set path to the stage param" in {
    val stage = estimator().setInput(feature)
    stage.setStageSavePath("/test/path")
    stage.getStageSavePath() shouldBe Some("/test/path")
  }
  it should "allow set/get spark params on a wrapped stage" in {
    val sparkStage = new StandardScaler()
    val stage = estimator(sparkMlStageIn = Some(sparkStage)).setInput(feature)
    stage.getSparkMlStage() shouldBe Some(sparkStage)
    for {
      sparkStage <- stage.getSparkMlStage()
      withMean = sparkStage.getOrDefault(sparkStage.withMean)
    } {
      withMean shouldBe false
      sparkStage.set[Boolean](sparkStage.withMean, true)
      sparkStage.get(sparkStage.withMean) shouldBe Some(true)
    }
  }

}
