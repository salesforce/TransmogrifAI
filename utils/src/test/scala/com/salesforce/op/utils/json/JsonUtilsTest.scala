/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.utils.json

import com.fasterxml.jackson.databind.annotation.JsonDeserialize
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks
import org.scalatest.{Assertion, PropSpec}

import scala.util.{Failure, Success}

@RunWith(classOf[JUnitRunner])
class JsonUtilsTest extends PropSpec with PropertyChecks with TestCommon {

  val bools = Gen.oneOf(true, false)
  val doubles = Gen.choose(Double.MinValue, Double.MaxValue)
  val longs = Gen.choose(Long.MinValue, Long.MaxValue)

  val dataGen =
    for {
      d <- doubles
      l <- longs
      b <- bools
    } yield {
      val t = TestDouble(d, Array.fill(5)(d), Seq.fill(3)(d), Map(1 -> Seq.fill(5)(l), 2 -> Seq.fill(2)(l)), None)
      if (b) {
        val nest = TestDouble(d, Array.fill(5)(d), Seq.fill(3)(d), Map(1 -> Seq.fill(5)(l), 2 -> Seq.fill(2)(l)), None)
        t.copy(nested = Some(nest))
      } else t
    }

  val specialDoubles = Gen.atLeastOne(Seq(
    0.0, 1.1, -2.2, Double.MaxValue, Double.MinValue, Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity)
  )
  val specialLongs = Gen.atLeastOne(Seq(0L, -1L, 2L, Long.MaxValue, Long.MinValue))

  val specialDataGen =
    for {
      d <- specialDoubles
      l <- specialLongs
      b <- bools
    } yield {
      val t = TestDouble(d.head, d.tail.toArray, d.tail.reverse, Map(1 -> Seq(l.head), 2 -> l.tail), None)
      if (b) {
        val nest = TestDouble(d.head, d.tail.toArray, d.tail.reverse, Map(1 -> Seq(l.head), 2 -> l.tail), None)
        t.copy(nested = Some(nest))
      } else t
    }

  property("handle random entries correctly") {
    forAll(dataGen)(checkJson)
    forAll(dataGen)(checkYaml)
  }

  property("handle special entries correctly") {
    forAll(specialDataGen)(checkJson)
    forAll(specialDataGen)(checkYaml)

  }

  property("handle empty collections correctly") {
    forAll(doubles) { d => checkJson(TestDouble(d, Array.empty, Seq.empty, Map.empty, None)) }
    forAll(doubles) { d => checkYaml(TestDouble(d, Array.empty, Seq.empty, Map.empty, None)) }
  }

  property("read json file") {
    val jsonFile = resourceFile(name = "Person.json")
    val person = Person("Foo Bar", Address("Talkeetna", "AK"))
    JsonUtils.fromFile[Person](jsonFile) match {
      case Failure(e) => fail(e)
      case Success(value) => value shouldBe person
    }
  }

  property("read yml file") {
    val ymlFile = resourceFile(name = "Person.yml")
    val person = Person("Foo Bar", Address("Talkeetna", "AK"))
    JsonUtils.fromFile[Person](ymlFile) match {
      case Failure(e) => fail(e)
      case Success(value) => value shouldBe person
    }
  }

  def checkJson(data: TestDouble): Unit = {
    val jsonValue: String = JsonUtils.toJsonString(data)
    JsonUtils.fromString[TestDouble](jsonValue) match {
      case Failure(e) => fail(e)
      case Success(r) => assert(r, data)
    }
  }

  def checkYaml(data: TestDouble): Unit = {
    val yamlValue: String = JsonUtils.toYamlString(data)
    JsonUtils.fromString[TestDouble](yamlValue) match {
      case Failure(e) => fail(e)
      case Success(r) => assert(r, data)
    }
  }

  def assert(v: TestDouble, expected: TestDouble): Assertion = {
    assert(v.v, expected.v)
    assert(v.seq, expected.seq)
    assert(v.arr, expected.arr)
    // TODO: re-enable; there are quotes in Int keys after Jackson upgrade
    // v.map shouldEqual expected.map
    for {
      v1 <- v.nested
      exp1 <- expected.nested
    } yield assert(v1, exp1)
    v.nested.isEmpty shouldBe expected.nested.isEmpty
  }

  def assert(v: Double, expected: Double): Assertion =
    if (v.isNaN) v.isNaN shouldBe expected.isNaN else v shouldBe expected

  def assert(v: Seq[Double], expected: Seq[Double]): Assertion = {
    v.seq.zip(expected).foreach { case (v1, v2) => assert(v1, v2) }
    v.seq.size shouldBe expected.size
  }

}

case class TestDouble
(
  v: Double,
  arr: Array[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  seq: Seq[Double],
  @JsonDeserialize(keyAs = classOf[java.lang.Integer])
  map: Map[Int, Seq[Long]],
  nested: Option[TestDouble]
)

case class Person(name: String, address: Address)

case class Address(city: String, state: String)
