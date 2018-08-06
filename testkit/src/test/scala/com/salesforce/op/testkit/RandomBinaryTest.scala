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

package com.salesforce.op.testkit

import com.salesforce.op.features.types.Binary
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.language.postfixOps


@RunWith(classOf[JUnitRunner])
class RandomBinaryTest extends FlatSpec with TestCommon {
  val numTries = 1000000
  val rngSeed = 12345

  private def truthWithProbability(probabilityOfTrue: Double) = {
    RandomBinary(probabilityOfTrue)
  }

  Spec[RandomBinary] should "generate empties, truths and falses" in {
    check(truthWithProbability(0.5) withProbabilityOfEmpty 0.5)
    check(truthWithProbability(0.3) withProbabilityOfEmpty 0.65)
    check(truthWithProbability(0.0) withProbabilityOfEmpty 0.1)
    check(truthWithProbability(1.0) withProbabilityOfEmpty 0.0)
  }

  private def check(g: RandomBinary) = {
    g reset rngSeed
    val numberOfEmpties = g limit numTries count (_.isEmpty)
    val expectedNumberOfEmpties = g.probabilityOfEmpty * numTries
    withClue(s"numEmpties = $numberOfEmpties, expected $expectedNumberOfEmpties") {
      math.abs(numberOfEmpties - expectedNumberOfEmpties) < numTries / 100 shouldBe true
    }

    val expectedNumberOfTruths = g.probabilityOfSuccess * (1 - g.probabilityOfEmpty) * numTries
    val numberOfTruths = g limit numTries count (Binary(true) ==)
    withClue(s"numTruths = $numberOfTruths, expected $expectedNumberOfTruths") {
      math.abs(numberOfTruths - expectedNumberOfTruths) < numTries / 100 shouldBe true
    }
  }
}
