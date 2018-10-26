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

package com.salesforce.op.utils.spark

import com.salesforce.op.test.TestCommon
import org.apache.spark.sql.types.{MetadataBuilder, MetadataWrapper}
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FlatSpec


@RunWith(classOf[JUnitRunner])
class RichMetadataTest extends FlatSpec with TestCommon {

  import com.salesforce.op.utils.spark.RichMetadata._

  private val map1 = Map(
    "1" -> 1L, "2" -> 1.0, "3" -> true, "4" -> "1",
    "5" -> Array(1L), "6" -> Array(1.0), "6" -> Array(x = true), "7" -> Array("1"),
    "8" -> Seq(1L), "9" -> Seq(1.0), "10" -> Seq(true), "11" -> Seq("1")
  )

  private val map2 = Map(
    "1" -> 1L, "2" -> 1.0, "3" -> false, "4" -> "2",
    "5" -> Array(1L, 2L), "6" -> Array(x = true), "7" -> Array("1"), "8" -> Seq(1L), "9" -> Seq(1.0),
    "10" -> Seq(true), "12" -> "12", "13" -> 11.5f
  )

  private val meta1 = map1.toMetadata

  implicit val formats: DefaultFormats = DefaultFormats

  Spec[RichMetadata] should "create a metadata from a map" in {
    meta1.json shouldBe Serialization.write(map1)
  }

  it should "throw an error on unsupported type in a map" in {
    intercept[RuntimeException](Map("a" -> TestClass("test")).toMetadata).getMessage shouldBe
      "Key 'a' has unsupported value TestClass(test) of type com.salesforce.op.utils.spark.TestClass"
  }

  it should "create a MetaDataWrapper with non empty map from a metadata " in {
    val wrap = meta1.wrapped
    meta1.underlyingMap shouldBe wrap.underlyingMap
    wrap shouldBe a[MetadataWrapper]
    meta1.isEmpty shouldBe false
  }

  it should "deep merge metadata correctly" in {
    val mergedMap = Map(
      "1" -> 2L, "2" -> 2.0, "3" -> true, "4" -> "12",
      "5" -> Array(1L, 1L, 2L), "6" -> Array(true, true), "7" -> Array("1", "1"),
      "8" -> Seq(1L, 1), "9" -> Seq(1.0, 1.0f), "10" -> Seq(true, true), "11" -> Seq("1"), "12" -> "12",
      "13" -> 11.5f
    )

    val mergedMetadata = meta1.deepMerge(map2.toMetadata)
    mergedMetadata.json shouldBe Serialization.write(mergedMap)
    mergedMetadata.prettyJson shouldBe Serialization.writePretty(mergedMap)
  }

  it should "deep merge the Array of metadata" in {
    val m1 = Map("1" -> Array(Map("val" -> "a").toMetadata)).toMetadata
    val m2 = Map("1" -> Array(Map("val" -> "b").toMetadata)).toMetadata
    val mergedValue = Map("1" -> Array(Map("val" -> "a").toMetadata, Map("val" -> "b").toMetadata)).toMetadata

    m1.deepMerge(m2) shouldBe mergedValue
  }

  it should "throw an error on incompatible value types in deep merge" in {
    the[RuntimeException] thrownBy meta1.deepMerge(Map("1" -> "test").toMetadata)
  }

  it should "be true on the same map when compared using deep equals" in {
    meta1.deepEquals(meta1) shouldBe true
  }

  it should "be false on different maps when compared using deep equals" in {
    meta1.deepEquals(map2.toMetadata) shouldBe false
  }

  it should "turn a metadata into summary metadata by putting it behind the summary key" in {
    val summaryMeta = meta1.toSummaryMetadata()
    summaryMeta.containsSummaryMetadata shouldBe true
  }

  it should "create summary for a given metadata" in {
    meta1.containsSummaryMetadata() shouldBe false

    // create a summary for the metadata.
    val expectedSummaryMetadata = Map("s" -> "summaryTest").toMetadata
    val richMetaDataWithSummary = meta1.withSummaryMetadata(expectedSummaryMetadata)

    richMetaDataWithSummary.getSummaryMetadata() shouldBe expectedSummaryMetadata
    richMetaDataWithSummary.containsSummaryMetadata() shouldBe true
  }

  it should "ignore empty values in maps" in {
    Map("None" -> None, "NULL" -> null).toMetadata shouldBe new MetadataBuilder().build()
    Map("None" -> None, "NULL" -> null, "OKKey" -> "ok!", "SomeKey" -> Some("ok")).toMetadata shouldBe
      new MetadataBuilder().putString("OKKey", "ok!").putString("SomeKey", "ok").build()
  }
}

case class TestClass(name: String)
