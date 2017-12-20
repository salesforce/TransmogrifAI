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
import scala.util.Random


@RunWith(classOf[JUnitRunner])
class RandomStreamTest extends FlatSpec with TestCommon {

  Spec[RandomStream[_]] should "apply" in {
    val rnd = new Random
    var i = 0
    val producer: Random => String = _ => {
      i += 1
      "hello " + i
    }
    val sut = new RandomStream[String](producer)
    val stream = sut(rnd)
    stream.next shouldBe "hello 1"
    stream.next shouldBe "hello 2"
    stream.next shouldBe "hello 3"
  }

  it should "map" in {
    val rnd = new Random
    var i = 0
    val producer: Random => Int = _ => {
      i += 1
      i
    }
    val src = new RandomStream[Int](producer)
    val sut = src map ("hello " +)
    val stream = sut(rnd)
    stream.next shouldBe "hello 1"
    stream.next shouldBe "hello 2"
    stream.next shouldBe "hello 3"
  }

  it should "do of" in {
    val rnd = new Random
    rnd.setSeed(123451)
    val sut = RandomStream of List("R", "G", "B")
    sut(rnd) shouldBe "B"
    sut(rnd) shouldBe "R"
    sut(rnd) shouldBe "R"
  }

  it should "distribute as requested" in {
    val rnd = new Random
    rnd.setSeed(123451)
    val sut = RandomStream of List("R", "G", "B") distributedAs List(0.5, 0.9, 1.0)
    val sample = 0 to 1000 map (_ => sut(rnd)) toList
    import org.scalatest.Matchers.{between => some}
    some(450, 550, sample) shouldBe "R"
    some(350, 450, sample) shouldBe "G"
    some(80, 120, sample) shouldBe "B"
  }

  it should "error when elements size does not match distribution" in {
    assertThrows[IllegalArgumentException] {
      RandomStream of List("R", "G") distributedAs List(0.5, 0.9, 1.0)
    }
  }

  it should "do trueWithProbability" in {
    val twp = RandomStream trueWithProbability 0.7
    val rnd = new Random
    rnd.setSeed(123452)
    (0 until 7 map (_ => twp(rnd)) toList) shouldBe List(true, false, false, true, false, true, true)
  }

  it should "do ofBits" in {
    val rnd = new Random
    rnd.setSeed(123452)
    val sut = RandomStream ofBits 0.7

    val stream = sut(rnd)
    (stream limit 5) shouldBe List(true, false, false, true, false)

    val stream2 = RandomStream ofBits(42L, 0.7)
    (stream2 limit 5) shouldBe List(false, true, true, true, true)
  }

  it should "do ofLongs" in {
    val rnd = new Random
    rnd.setSeed(123452)
    val sut = RandomStream ofLongs

    val stream = sut(rnd)
    (stream limit 5) shouldBe List(
      7682038666078246895L, -2087874339651216632L, -5136527809831860773L, -8419421625251191413L, -4688129678144181882L)
  }

  it should "do ofLongs with range" in {
    val rnd = new Random
    rnd.setSeed(123452)
    val sut = RandomStream ofLongs(-25, 25)

    val stream = sut(rnd)
    (stream limit 5) shouldBe
      List(20, -7, 2, 12, -7)
  }

  it should "do RandomBetween" in {
    val rnd = new Random
    rnd.setSeed(123457)
    RandomStream.randomBetween(10, 0)(rnd) shouldBe 10
    RandomStream.randomBetween(10, 0)(rnd) shouldBe 10
    RandomStream.randomBetween(0, 10)(rnd) shouldBe 9
    RandomStream.randomBetween(0, 10)(rnd) shouldBe 7
    RandomStream.randomBetween(0, 10)(rnd) shouldBe 7
    RandomStream.randomBetween(0, 10)(rnd) shouldBe 1
    RandomStream.randomBetween(0, 10)(rnd) shouldBe 5
  }

  it should "do ofChunks" in {
    val rnd = new Random
    rnd.setSeed(123451)
    val src = RandomStream of List("R", "G", "B")
    val sut = RandomStream.randomChunks[String](3, 7)(src)
    sut(rnd).toList shouldBe List("R", "R", "R", "G")
    sut(rnd).toList shouldBe List("B", "B", "R", "B")
    sut(rnd).toList shouldBe List("G", "R", "B", "R", "B", "G")
  }
}
