/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
