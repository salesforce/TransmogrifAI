/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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

  def toBase64(b: Array[Byte]): String = new String(java.util.Base64.getEncoder.encode(b))
}
