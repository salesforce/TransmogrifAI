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

package com.salesforce.op.testkit

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec}

import scala.language.postfixOps


@RunWith(classOf[JUnitRunner])
class RandomSetTest extends FlatSpec with TestCommon with Assertions {
  private val numTries = 10000
  private val rngSeed = 314159214142136L

  private def check[D, T <: OPSet[D]](
    g: RandomSet[D, T],
    minLen: Int, maxLen: Int,
    predicate: (D => Boolean) = (_: D) => true
  ) = {
    g reset rngSeed

    def segment = g limit numTries

    segment count (_.value.size < minLen) shouldBe 0
    segment count (_.value.size > maxLen) shouldBe 0
    segment foreach (Set => Set.value foreach { x =>
      predicate(x) shouldBe true
    })
  }

  Spec[MultiPickList] should "generate multipicklists" in {
    val sut = RandomMultiPickList.of(RandomText.countries, maxLen = 5)

    check[String, MultiPickList](sut, 0, 5, _.nonEmpty)

    val expected = List(
      Set(),
      Set("Aldorria", "Palau", "Glubbdubdrib"),
      Set(),
      Set(),
      Set("Sweden", "Wuhu Islands", "Tuvalu")
    )

    {sut reset 42; sut limit 5 map (_.value)} shouldBe expected

    {sut reset 42; sut limit 5 map (_.value)} shouldBe expected
  }

}
