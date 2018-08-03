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

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks

@RunWith(classOf[JUnitRunner])
class URLTest extends PropSpec with PropertyChecks with TestCommon {

  val badOnes = Table("bad ones",
    None,
    Some(""),
    Some("protocol://domain.codomain"),
    Some("httpd://domain.codomain"),
    Some("http://domain."),
    Some("ftp://.codomain"),
    Some("https://.codomain"),
    Some("//domain.nambia"),
    Some("http://\u00ff\u0080\u007f\u0000.com") // scalastyle:off
  )

  val goodOnes = Table("good ones",
    "https://nothinghere.com?Eli=%E6%B8%87%40",
    "http://nothingthere.com?Chr=%E5%95%A9%E7%B1%85&Raj=%E7%B5%89%EC%AE%A1&Hir=%E5%B3%8F%E0%B4%A3",
    "ftp://my.red.book.com/amorcito.mio",
    "http://secret.gov?Cla=%E9%99%B9%E4%8A%93&Cha=%E3%95%98%EA%A3%A7&Eve=%EC%91%90%E8%87%B1",
    "ftp://nukes.mil?Lea=%E2%BC%84%EB%91%A3&Mur=%E2%83%BD%E1%92%83"
  )

  property("validate urls") {
    forAll(badOnes) {
      sample => URL(sample).isValid shouldBe false
    }
    forAll(goodOnes) {
      sample => URL(sample).isValid shouldBe true
    }
    forAll(goodOnes) {
      sample => URL(sample).isValid(protocols = Array("http")) shouldBe sample.startsWith("http:")
    }
  }

  property("extract domain") {
    val samples = Table("samples",
      "https://nothinghere.com?Eli=%E6%B8%87%40" -> "nothinghere.com",
      "http://nothingthere.com?Chr=%E5%85&Raj=%E7%B5%AE%A1&Hir=%8F%E0%B4%A3" -> "nothingthere.com",
      "ftp://my.red.book.com/amorcito.mio" -> "my.red.book.com",
      "http://secret.gov?Cla=%E9%99%B9%E4%8A%93&Cha=%E3&Eve=%EC%91%90%E8%87%B1" -> "secret.gov",
      "ftp://nukes.mil?Lea=%E2%BC%84%EB%91%A3&Mur=%E2%83%BD%E1%92%83" -> "nukes.mil"
    )

    URL(None).domain shouldBe None

    forAll(samples) {
      case (sample, expected) =>
        val url = URL(sample)
        val domain = url.domain
        domain shouldBe Some(expected)
    }
  }

  property("extract protocol") {
    val samples = Table("samples",
      "https://nothinghere.com?Eli=%E6%B8%87%40" -> "https",
      "http://nothingthere.com?Chr=%E5%85&Raj=%E7%B5%AE%A1&Hir=%8F%E0%B4%A3" -> "http",
      "ftp://my.red.book.com/amorcito.mio" -> "ftp"
    )

    URL(None).protocol shouldBe None

    forAll(samples) {
      case (sample, expected) =>
        val url = URL(sample)
        val domain = url.protocol
        domain shouldBe Some(expected)
    }
  }
}
