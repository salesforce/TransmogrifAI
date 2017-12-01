/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types.{Currency, Percent, Real, RealNN}
import com.salesforce.op.test.TestCommon
import com.salesforce.op.testkit.RandomReal._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RandomRealTest extends FlatSpec with TestCommon {
  val numTries = 1000000

  /**
   * TODO: We have a type problem here: need Real, get Nothing. Should use the solution from
   * [[http://www.cakesolutions.net/teamblogs/default-type-parameters-with-implicits-in-scala]]
   */
  // ignore should "cast to default data type" in {
  //  check(uniform(1.0, 2.0), probabilityOfEmpty = 0.5, range = (1.0, 2.0))
  // }

  Spec[RandomReal[Real]]  should "Give Normal distribution with mean 1 sigma 0.1, 10% nulls" in {
    val normalReals = normal[Real](1.0, 0.2)
    check(normalReals, probabilityOfEmpty = 0.1, range = (-2.0, 4.0))
  }

  Spec[RandomReal[Real]] should "Give Uniform distribution on 1..2, half nulls" in {
    check(uniform[Real](1.0, 2.0), probabilityOfEmpty = 0.5, range = (1.0, 2.0))
  }

  it should "Give Poisson distribution with mean 4, 20% nulls" in {
    check(poisson[Real](4.0), probabilityOfEmpty = 0.2, range = (0.0, 15.0))
  }

  it should "Give Exponential distribution with mean 1, 1% nulls" in {
    check(exponential[Real](1.0), probabilityOfEmpty = 0.01, range = (0.0, 15.0))
  }

  it should "Give Gamma distribution with mean 5, 0% nulls" in {
    check(gamma[Real](5.0), probabilityOfEmpty = 0.0, range = (0.0, 25.0))
  }

  it should "Give LogNormal distribution with mean 0.25, 20% nulls" in {
    check(logNormal[Real](0.25, 0.001), probabilityOfEmpty = 0.7, range = (0.1, 15.0))
  }

  it should "Weibull distribution (4.0, 5.0), 20% nulls" in {
    check(weibull[Real](4.0, 5.0), probabilityOfEmpty = 0.2, range = (0.0, 15.0))
  }

  Spec[RandomReal[RealNN]] should "give no nulls" in {
    check(normal[RealNN](1.0, 0.2), probabilityOfEmpty = 0.0, range = (-2.0, 4.0))
  }

  Spec[RandomReal[Currency]] should "distribute money normally" in {
    check(normal[Currency](1.0, 0.2), probabilityOfEmpty = 0.5, range = (-2.0, 4.0))
  }

  Spec[RandomReal[Percent]] should "distribute percentage evenly" in {
    check(uniform[Percent](1.0, 2.0), probabilityOfEmpty = 0.5, range = (0.0, 2.0))
  }

  private val rngSeed = 7688721

  private def check[T <: Real](
    src: RandomReal[T],
    probabilityOfEmpty: Double,
    range: (Double, Double)) = {
    val sut = src withProbabilityOfEmpty probabilityOfEmpty
    sut reset rngSeed

    val found = sut.next
    sut reset rngSeed
    val foundAfterReseed = sut.next
    if (foundAfterReseed != found) {
      sut.reset(rngSeed)
    }
    withClue(s"generator reset did not work for $sut") {
      foundAfterReseed shouldBe found
    }
    sut reset rngSeed

    val numberOfNulls = sut limit numTries count (_.isEmpty)

    val expectedNumberOfNulls = probabilityOfEmpty * numTries
    withClue(s"numNulls = $numberOfNulls, expected $expectedNumberOfNulls") {
      math.abs(numberOfNulls - expectedNumberOfNulls) < numTries / 100 shouldBe true
    }

    val numberOfOutliers = sut limit numTries count (xOpt => xOpt.value.exists(x => x < range._1 || x > range._2))

    numberOfOutliers should be < (numTries / 1000)

  }
}
