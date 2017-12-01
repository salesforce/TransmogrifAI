/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec}

import scala.language.postfixOps


@RunWith(classOf[JUnitRunner])
class RandomSetTest extends FlatSpec with TestCommon with Assertions {
  private val numTries = 10000
  private val rngSeed = 314159214142136L

  private def check[D, T <: OPSet[D]](
    g: RandomSet[D, T],
    minLen: Int, maxLen: Int,
    predicate: (D => Boolean) = (_: D) => true
  ) = {
    g reset rngSeed

    def segment = g limit numTries

    segment count (_.value.size < minLen) shouldBe 0
    segment count (_.value.size > maxLen) shouldBe 0
    segment foreach (Set => Set.value foreach { x =>
      predicate(x) shouldBe true
    })
  }

  Spec[MultiPickList] should "generate multipicklists" in {
    val sut = RandomMultiPickList.of(RandomText.countries, maxLen = 5)

    check[String, MultiPickList](sut, 0, 5, _.nonEmpty)

    val expected = List(
      Set(),
      Set("Aldorria", "Palau", "Glubbdubdrib"),
      Set(),
      Set(),
      Set("Sweden", "Wuhu Islands", "Tuvalu")
    )

    {sut reset 42; sut limit 5 map (_.value)} shouldBe expected

    {sut reset 42; sut limit 5 map (_.value)} shouldBe expected
  }

}
