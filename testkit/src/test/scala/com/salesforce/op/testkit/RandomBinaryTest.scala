/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types.Binary
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.language.postfixOps


@RunWith(classOf[JUnitRunner])
class RandomBinaryTest extends FlatSpec with TestCommon {
  val numTries = 1000000
  val rngSeed = 12345

  private def truthWithProbability(probabilityOfTrue: Double) = {
    RandomBinary(probabilityOfTrue)
  }

  Spec[RandomBinary] should "generate empties, truths and falses" in {
    check(truthWithProbability(0.5) withProbabilityOfEmpty 0.5)
    check(truthWithProbability(0.3) withProbabilityOfEmpty 0.65)
    check(truthWithProbability(0.0) withProbabilityOfEmpty 0.1)
    check(truthWithProbability(1.0) withProbabilityOfEmpty 0.0)
  }

  private def check(g: RandomBinary) = {
    g reset rngSeed
    val numberOfEmpties = g limit numTries count (_.isEmpty)
    val expectedNumberOfEmpties = g.probabilityOfEmpty * numTries
    withClue(s"numEmpties = $numberOfEmpties, expected $expectedNumberOfEmpties") {
      math.abs(numberOfEmpties - expectedNumberOfEmpties) < numTries / 100 shouldBe true
    }

    val expectedNumberOfTruths = g.probabilityOfSuccess * (1 - g.probabilityOfEmpty) * numTries
    val numberOfTruths = g limit numTries count (Binary(true) ==)
    withClue(s"numTruths = $numberOfTruths, expected $expectedNumberOfTruths") {
      math.abs(numberOfTruths - expectedNumberOfTruths) < numTries / 100 shouldBe true
    }
  }
}
