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
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class EnumEntrySerializerTest extends FlatSpec with TestCommon {

  implicit val formats = DefaultFormats + EnumEntrySerializer[TestEnumType](TestEnumType)

  val data = TestData(a = TestEnumType.One, b = Seq(TestEnumType.Two, TestEnumType.Three))
  val dataJson = """{"a":"One","b":["Two","Three"]}"""

  Spec(EnumEntrySerializer.getClass) should "write enum entries" in {
    compact(Extraction.decompose(data)) shouldBe dataJson
  }
  it should "read enum entries" in {
    parse(dataJson).extract[TestData] shouldBe data
  }
  it should "read enum entries ignoring case" in {
    parse(dataJson.toLowerCase).extract[TestData] shouldBe data
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
