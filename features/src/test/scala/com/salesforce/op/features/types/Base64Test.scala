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

package com.salesforce.op.features.types

import java.nio.charset.Charset

import com.salesforce.op.test.TestCommon
import org.apache.commons.io.IOUtils
import org.junit.runner.RunWith
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks

@RunWith(classOf[JUnitRunner])
class Base64Test extends PropSpec with PropertyChecks with TestCommon {

  property("handle empty") {
    forAll(None) {
      (v: Option[String]) =>
        Base64(v).asBytes shouldBe None
        Base64(v).asString shouldBe None
        Base64(v).asInputStream shouldBe None
    }
  }

  property("can show byte contents") {
    forAll {
      (b: Array[Byte]) =>
        val b64 = toBase64(b)
        (Base64(b64).asBytes map (_.toList)) shouldBe Some(b.toList)
    }
  }

  property("can show string contents") {
    forAll {
      (s: String) =>
        val b64 = toBase64(s.getBytes)
        Base64(b64).asString shouldBe Some(s)
    }
  }

  property("produce a stream") {
    forAll {
      (s: String) =>
        val b64 = toBase64(s.getBytes)
        Base64(b64).asInputStream.map(IOUtils.toString(_, Charset.defaultCharset())) shouldBe Some(s)
    }
  }

  property("produce a stream and map over it") {
    forAll {
      (s: String) =>
        val b64 = toBase64(s.getBytes)
        Base64(b64).mapInputStream(IOUtils.toString(_, Charset.defaultCharset())) shouldBe Some(s)
    }
  }

  def toBase64(b: Array[Byte]): String = new String(java.util.Base64.getEncoder.encode(b))
}
