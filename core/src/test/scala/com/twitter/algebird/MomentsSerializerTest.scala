package com.twitter.algebird

import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.Serialization
import org.junit.runner.RunWith
import org.scalatest.Matchers._
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class MomentsSerializerTest extends FlatSpec {
  val moments = Moments(0L, 1.0, 2.0, 3.0, 4.0)
  val momentsApply1 = Moments(0L)
  val momentsApply2 = Moments(0L, 1L, 2L, 3L, 4L)

  val momentsJson = """{"m0":0,"m1":1.0,"m2":2.0,"m3":3.0,"m4":4.0}"""
  val momentsApply1Json = """{"m0":1,"m1":0.0,"m2":0.0,"m3":0.0,"m4":0.0}"""

  implicit val formats: Formats = DefaultFormats + new MomentsSerializer

  it should "properly serialize the Moments class regardless of apply method used" in {

    Serialization.write[Moments](moments) shouldBe momentsJson
    Serialization.write[Moments](momentsApply1) shouldBe momentsApply1Json
    Serialization.write[Moments](momentsApply2) shouldBe momentsJson
  }

  it should "properly deserialize the Moments class" in {
    Serialization.read[Moments]{momentsJson} shouldBe moments
    Serialization.read[Moments]{momentsApply1Json} shouldBe momentsApply1
  }

  it should "recover the original class after a serialization/deserialization round-trip" in {
    Serialization.read[Moments]{Serialization.write[Moments](moments)} shouldBe moments
  }
}
