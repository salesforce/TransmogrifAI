/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import java.text.SimpleDateFormat

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec}

import scala.language.postfixOps

@RunWith(classOf[JUnitRunner])
class RandomIntegralTest extends FlatSpec with TestCommon with Assertions {
  private val numTries = 10000
  private val rngSeed = 314159214142135L

  private def check[T <: Integral](
    g: RandomIntegral[T],
    predicate: Long => Boolean = _ => true
  ) = {
    g reset rngSeed

    def segment = g limit numTries

    val numberOfEmpties = segment count (_.isEmpty)

    val expectedNumberOfEmpties = g.probabilityOfEmpty * numTries

    withClue(s"numEmpties = $numberOfEmpties, expected $expectedNumberOfEmpties") {
      math.abs(numberOfEmpties - expectedNumberOfEmpties) < 2 * math.sqrt(numTries) shouldBe true
    }

    val maybeValues = segment filterNot (_.isEmpty) map (_.value)
    val values = maybeValues collect { case Some(s) => s }

    values foreach (x => predicate(x) shouldBe true)

    withClue(s"number of distinct values = ${values.size}, expected:") {
      math.abs(maybeValues.size - values.toSet.size) < maybeValues.size / 20
    }

  }

  private val df = new SimpleDateFormat("dd/MM/yy")

  Spec[RandomIntegral[Integral]] should "generate empties and distinct numbers" in {
    val sut0 = RandomIntegral.integrals
    val sut = sut0.withProbabilityOfEmpty(0.3)
    check(sut)
    sut.probabilityOfEmpty shouldBe 0.3
  }

  Spec[RandomIntegral[Integral]] should "generate empties and distinct numbers in some range" in {
    val sut0 = RandomIntegral.integrals(100, 200)
    val sut = sut0.withProbabilityOfEmpty(0.3)
    check(sut)
    sut.probabilityOfEmpty shouldBe 0.3
  }

  Spec[RandomIntegral[Date]] should "generate dates" in {
    val sut = RandomIntegral.dates(df.parse("01/01/2017"), 1000, 1000000)
    var d0 = 0L
    check(sut withProbabilityOfEmpty 0.01, d => {
      val d1 = d0
      d0 = d
      d0 > d1
    })
  }

  Spec[RandomIntegral[DateTime]] should "generate dates with times" in {
    val sut = RandomIntegral.datetimes(df.parse("08/24/2017"), 1000, 1000000)
    var d0 = 0L
    check(sut withProbabilityOfEmpty 0.001, d => {
      val d1 = d0
      d0 = d
      d0 > d1
    })
  }
}
