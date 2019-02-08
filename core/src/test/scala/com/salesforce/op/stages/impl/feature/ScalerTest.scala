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

import com.salesforce.op.test.TestSparkContext
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ScalerTest extends FlatSpec with TestSparkContext{

  Spec[Scaler] should "error on invalid data" in {
    val error = intercept[IllegalArgumentException](
      Scaler.apply(scalingType = ScalingType.Linear, args = EmptyArgs())
    )
    error.getMessage shouldBe "Invalid combination of scaling type 'Linear' and args type 'EmptyArgs'"
  }

  it should "correctly build construct a LinearScaler" in {
    val linearScaler = Scaler.apply(scalingType = ScalingType.Linear,
      args = LinearScalerArgs(slope = 1.0, intercept = 2.0))
    linearScaler shouldBe a[LinearScaler]
    linearScaler.scalingType shouldBe ScalingType.Linear
  }

  it should "correctly build construct a LogScaler" in {
    val linearScaler = Scaler.apply(scalingType = ScalingType.Logarithmic, args = EmptyArgs())
    linearScaler shouldBe a[LogScaler]
    linearScaler.scalingType shouldBe ScalingType.Logarithmic
  }
}

