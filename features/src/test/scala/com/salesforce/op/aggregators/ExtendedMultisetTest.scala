/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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

