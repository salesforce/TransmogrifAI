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

package com.salesforce.op

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class UIDTest extends FlatSpec with TestCommon {

  Spec(UID.getClass) should "generate UIDs" in {
    (1 to 100000).map(_ => UID[UIDTest]).toSet.size shouldBe 100000
  }

  it should "allow counting UIDs" in {
    val start = UID.count()
    (1 to 100).foreach(_ => UID[UIDTest])
    val end = UID.count()
    end - start shouldBe 100
  }

  it should "allow reset UIDs to a specific count" in {
    val count = UID.count()
    val first = (1 to 100).map(_ => UID[UIDTest])
    UID.reset(count)
    val second = (1 to 100).map(_ => UID[UIDTest])
    first should contain theSameElementsAs second
    UID.reset()[UIDTest] shouldBe "UIDTest_000000000001"
  }

  it should "allow reset UIDs" in {
    UID.reset()
    val first = (1 to 100).map(_ => UID[UIDTest])
    UID.reset()
    val second = (1 to 100).map(_ => UID[UIDTest])
    first should contain theSameElementsAs second
  }

  it should "parse from string" in {
    UID.reset().fromString(UID[UIDTest]) shouldBe ("UIDTest", "000000000001")
  }

  it should "error on invalid string" in {
    intercept[IllegalArgumentException](UID.fromString("foo")).getMessage shouldBe "Invalid UID: foo"
  }
}
