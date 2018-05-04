/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.json

import com.salesforce.op.test.TestCommon
import enumeratum.{Enum, EnumEntry}
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, Extraction}
import org.junit.runner.RunWith
import org.scalatest.FunSpec
import org.scalatest.junit.JUnitRunner

import scala.util.Success


@RunWith(classOf[JUnitRunner])
class EnumEntrySerializerTest extends FunSpec with TestCommon {

  implicit val formats = DefaultFormats + EnumEntrySerializer.json4s[TestEnumType](TestEnumType)
  val serdes = Seq(EnumEntrySerializer.jackson[TestEnumType](TestEnumType))

  val data = TestData(a = TestEnumType.One, b = Seq(TestEnumType.Two, TestEnumType.Three))
  val dataJson = """{"a":"One","b":["Two","Three"]}"""

  describe("EnumEntrySerializer") {
    describe("(json4s)") {
      it("write enum entries") {
        compact(Extraction.decompose(data)) shouldBe dataJson
      }
      it("read enum entries") {
        parse(dataJson).extract[TestData] shouldBe data
      }
      it("read enum entries ignoring case") {
        parse(dataJson.toLowerCase).extract[TestData] shouldBe data
      }
    }
    describe("(jackson)") {
      it("write enum entries") {
        JsonUtils.toJsonString(data, pretty = false, serdes = serdes) shouldBe dataJson
      }
      it("read enum entries") {
        JsonUtils.fromString[TestData](dataJson, serdes = serdes) shouldBe Success(data)
      }
      it("read enum entries ignoring case") {
        JsonUtils.fromString[TestData](dataJson.toLowerCase, serdes = serdes) shouldBe Success(data)
      }
    }
  }

}

private[json] case class TestData(a: TestEnumType, b: Seq[TestEnumType])

sealed trait TestEnumType extends EnumEntry with Serializable
object TestEnumType extends Enum[TestEnumType] {
  val values = findValues
  case object One extends TestEnumType
  case object Two extends TestEnumType
  case object Three extends TestEnumType
}
