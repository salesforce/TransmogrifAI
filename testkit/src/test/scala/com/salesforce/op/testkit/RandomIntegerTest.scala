package com.salesforce.op.testkit

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec}


@RunWith(classOf[JUnitRunner])
class RandomIntegerTest extends FlatSpec with TestCommon with Assertions {
  it should "produce integers within defined range" in {
    val from = 1
    val to = 500
    val n = 5000
    val randomInts = RandomInteger.integers(from, to).take(n).toSeq
    val results = randomInts.map(i => i.value.get < to & i.value.get >= from)

    assert(randomInts.map(x => !x.value.get.isInstanceOf[Int]).toSet.contains(false))
    assert(!results.toSet.contains(false))
  }

}
