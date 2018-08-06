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
import org.apache.spark.sql.types.{MetadataBuilder, StructField}
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class RichMetadataTest extends FlatSpec with TestCommon {

  import com.salesforce.op.utils.spark.RichMetadata._

  Spec[RichMetadata] should "create a metadata from a map" in {
    val expected = Map(
      "1" -> 1L, "2" -> 1.0, "3" -> true, "4" -> "1",
      "5" -> Array(1L), "6" -> Array(1.0), "6" -> Array(true), "7" -> Array("1"),
      "8" -> Seq(1L), "9" -> Seq(1.0), "10" -> Seq(true), "11" -> Seq("1")
    )
    val meta = expected.toMetadata
    implicit val formats = DefaultFormats
    meta.json shouldBe Serialization.write(expected)
  }

  it should "throw an error on unsupported type in a map" in {
    the[RuntimeException] thrownBy Map("a" -> Map("b" -> 1)).toMetadata
  }

}
