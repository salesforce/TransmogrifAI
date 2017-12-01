/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class ScalaStyleValidationTest extends FlatSpec with Matchers with Assertions {
  import scala.Throwable
  private def +(x: Int, y: Int) = x + y
  private def -(x: Int, y: Int) = x - y
  private def *(x: Int, y: Int) = x * y
  private def /(x: Int, y: Int) = x / y
  private def +-(x: Int, y: Int) = x + (-y)
  private def xx_=(y: Int) = println(s"setting xx to $y")

  "bad names" should "never happen" in {
    "def _=abc = ???" shouldNot compile
    true shouldBe true
  }

  "non-ascii" should "not be allowed" in {
//    "def ⇒ = ???" shouldNot compile // it does not even compile as a string
  }

}
