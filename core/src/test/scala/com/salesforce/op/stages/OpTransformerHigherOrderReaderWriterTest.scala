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

import com.salesforce.op.dsl.RichNumericFeature
import com.salesforce.op.features.types._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpTransformerHigherOrderReaderWriterTest extends OpPipelineStageReaderWriterTest with RichNumericFeature {

  import OpTransformerHigherOrderReaderWriterTest._

  val v1: Int = 10
  val v2: Double = 5.5
  val v3: Long = 300L

  val stage: OpPipelineStageBase = height.map(
    fnc0(v1, v2, v3),
    lambdaCtorArgs = Array(Int.box(v1), Double.box(v2), Long.box(v3)
    )
  ).originStage

  override val hasOutputName = false

  val expected = Array(3935.5.toReal, 305.5.toReal, 305.5.toReal, 305.5.toReal, 2165.5.toReal, 1985.5.toReal)
}

object OpTransformerHigherOrderReaderWriterTest {
  private def fnc0(n: Int, v2: Double, v3: Long) = (x: RealNN) => x.value.map(_ * n + v2 + v3).toReal
}
