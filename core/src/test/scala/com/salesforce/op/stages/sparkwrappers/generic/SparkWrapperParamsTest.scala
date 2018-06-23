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

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterEach, FlatSpec}

@RunWith(classOf[JUnitRunner])
class SparkWrapperParamsTest extends FlatSpec with BeforeAndAfterEach with TestCommon {

  var swEstimator: SwUnaryEstimator[Real, Real, StandardScalerModel, StandardScaler] = _

  override protected def beforeEach(): Unit = {
    swEstimator = new SwUnaryEstimator[Real, Real, StandardScalerModel, StandardScaler](
      inputParamName = "foo",
      outputParamName = "foo2",
      operationName = "foo3",
      sparkMlStageIn = None
    )
  }

  val path = "test"

  Spec[SparkWrapperParams[_]] should "when setting path, it should also set path to the stage param" in {
    swEstimator.setSavePath(path)

    swEstimator.getSavePath() shouldBe path
    swEstimator.getStageSavePath().get shouldBe swEstimator.getSavePath()
  }

  it should "have proper default values for path and stage" in {
    swEstimator.getSavePath() shouldBe ""
    swEstimator.getSparkMlStage() shouldBe None
  }

}
