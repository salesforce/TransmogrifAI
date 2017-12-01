/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest._
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class EventTest extends FlatSpec with TestCommon {

  Spec[Event[_]] should "compare" in {
    val sut1 = Event[Integral](123, Integral(42), isResponse = false)
    val sut2 = Event[Integral](321, Integral(666))
    (sut1 compare sut1) shouldBe 0
    (sut1 compare sut2) shouldBe -1
    (sut2 compare sut1) shouldBe 1
    (sut2 compare sut2) shouldBe 0
    sut2.isResponse shouldBe false
  }

}
