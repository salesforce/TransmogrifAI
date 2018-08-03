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

import java.text.SimpleDateFormat

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec}

import scala.language.postfixOps

@RunWith(classOf[JUnitRunner])
class RandomIntegralTest extends FlatSpec with TestCommon with Assertions {
  private val numTries = 10000
  private val rngSeed = 314159214142135L

  private def check[T <: Integral](
    g: RandomIntegral[T],
    predicate: Long => Boolean = _ => true
  ) = {
    g reset rngSeed

    def segment = g limit numTries

    val numberOfEmpties = segment count (_.isEmpty)

    val expectedNumberOfEmpties = g.probabilityOfEmpty * numTries

    withClue(s"numEmpties = $numberOfEmpties, expected $expectedNumberOfEmpties") {
      math.abs(numberOfEmpties - expectedNumberOfEmpties) < 2 * math.sqrt(numTries) shouldBe true
    }

    val maybeValues = segment filterNot (_.isEmpty) map (_.value)
    val values = maybeValues collect { case Some(s) => s }

    values foreach (x => predicate(x) shouldBe true)

    withClue(s"number of distinct values = ${values.size}, expected:") {
      math.abs(maybeValues.size - values.toSet.size) < maybeValues.size / 20
    }

  }

  private val df = new SimpleDateFormat("dd/MM/yy")

  Spec[RandomIntegral[Integral]] should "generate empties and distinct numbers" in {
    val sut0 = RandomIntegral.integrals
    val sut = sut0.withProbabilityOfEmpty(0.3)
    check(sut)
    sut.probabilityOfEmpty shouldBe 0.3
  }

  Spec[RandomIntegral[Integral]] should "generate empties and distinct numbers in some range" in {
    val sut0 = RandomIntegral.integrals(100, 200)
    val sut = sut0.withProbabilityOfEmpty(0.3)
    check(sut, i => i >= 100 && i < 200)
    sut.probabilityOfEmpty shouldBe 0.3
  }

  Spec[RandomIntegral[Date]] should "generate dates" in {
    val sut = RandomIntegral.dates(df.parse("01/01/2017"), 1000, 1000000)
    var d0 = 0L
    check(sut withProbabilityOfEmpty 0.01, d => {
      val d1 = d0
      d0 = d
      d0 > d1
    })
  }

  Spec[RandomIntegral[DateTime]] should "generate dates with times" in {
    val sut = RandomIntegral.datetimes(df.parse("08/24/2017"), 1000, 1000000)
    var d0 = 0L
    check(sut withProbabilityOfEmpty 0.001, d => {
      val d1 = d0
      d0 = d
      d0 > d1
    })
  }
}
