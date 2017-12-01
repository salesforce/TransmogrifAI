/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.language.postfixOps


@RunWith(classOf[JUnitRunner])
class InfiniteStreamTest extends FlatSpec with TestCommon {

  Spec[InfiniteStream[_]] should "map" in {
    var i = 0
    val src = new InfiniteStream[Int] {
      override def next: Int = {
        i += 1;
        i
      }
    }

    val sut = src map (5 +)

    while (i < 10) sut.next shouldBe (i + 5)
  }

}
