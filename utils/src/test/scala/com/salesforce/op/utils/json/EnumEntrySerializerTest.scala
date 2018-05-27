/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
