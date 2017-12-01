/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
