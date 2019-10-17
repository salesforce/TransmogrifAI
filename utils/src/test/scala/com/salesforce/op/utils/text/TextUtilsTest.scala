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

package com.salesforce.op.utils.text

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class TextUtilsTest extends FlatSpec with TestCommon {
  Spec(TextUtils.getClass) should "concat strings" in {
    TextUtils.concat("Left", "Right", ",") shouldBe "Left,Right"
  }

  it should "concat with no effect for right half alone" in {
    TextUtils.concat("", "Right", ",") shouldBe "Right"
  }

  it should "concat with no effect for left half alone" in {
    TextUtils.concat("Left", "", ",") shouldBe "Left"
  }

  it should "concat empty strings" in {
    TextUtils.concat("", "", ",") shouldBe ""
  }

  it should "clean a string with special chars by default" in {
    TextUtils.cleanString("A string wit#h %bad pun&ctuation mark<=>s") shouldBe "AStringWitHBadPunCtuationMarkS"
  }

  it should "clean an Option(string) with special chars by default" in {
    val testString: Option[String] = Some("A string wit#h %bad pun&ctuation mark<=>s")
    TextUtils.cleanOptString(testString) shouldBe Some("AStringWitHBadPunCtuationMarkS")
  }

  it should "ignore the case and not clean a string with punctuations " +
    "when cleanPunctuations=true & ignoreCase=true" in {
    val actual = TextUtils.cleanString("Salesforce.com", cleanTextParams = CleanTextParams(true, false))
    actual shouldBe "salesforce.com"
  }
}
