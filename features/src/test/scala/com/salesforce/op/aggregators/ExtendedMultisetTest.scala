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

package com.salesforce.op.aggregators

import com.salesforce.op.aggregators.{ExtendedMultiset => SUT}
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest._
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ExtendedMultisetTest extends FlatSpec with TestCommon {

  Spec[ExtendedMultiset] should "add" in {
    val sut1 = Map[String, Long]("a" -> 1, "b" -> 0, "c" -> 42)
    val sut2 = Map[String, Long]("d" -> 7, "b" -> 0, "c" -> 2)

    SUT.plus(sut1, sut2) shouldBe Map[String, Long]("a" -> 1, "c" -> 44, "d" -> 7)
    SUT.plus(SUT.zero, sut2) shouldBe sut2
    SUT.plus(sut1, SUT.zero) shouldBe sut1
  }

  it should "subtract" in {
    val sut1 = Map[String, Long]("a" -> 1, "b" -> 0, "c" -> 42)
    val sut2 = Map[String, Long]("d" -> 7, "b" -> 0, "c" -> 2)

    SUT.minus(sut1, sut2) shouldBe Map[String, Long]("a" -> 1, "c" -> 40, "d" -> -7)
    SUT.minus(sut1, SUT.zero) shouldBe Map[String, Long]("a" -> 1, "c" -> 42)
    SUT.minus(SUT.zero, sut2) shouldBe Map[String, Long]("d" -> -7, "c" -> -2)
  }
}

