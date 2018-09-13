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

package com.salesforce.op.cli.gen

import com.salesforce.op.cli.gen.AvroField._
import com.salesforce.op.test.TestCommon
import org.apache.avro.Schema
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec}

import scala.collection.JavaConverters._
import scala.language.postfixOps

/**
 * Test for AvroField methods.
 */
@RunWith(classOf[JUnitRunner])
class AvroFieldTest extends FlatSpec with TestCommon with Assertions {

  Spec[AvroField] should "do from" in {
    val types = List(
      Schema.Type.STRING,
      //  Schema.Type.BYTES, // somehow this avro type is not covered (yet)
      Schema.Type.INT,
      Schema.Type.LONG,
      Schema.Type.FLOAT,
      Schema.Type.DOUBLE,
      Schema.Type.BOOLEAN
    )
    val simpleSchemas = types map Schema.create

    val unions = List(
      Schema.createUnion((Schema.Type.NULL::Schema.Type.INT::Nil) map Schema.create asJava),
      Schema.createUnion((Schema.Type.INT::Schema.Type.NULL::Nil) map Schema.create asJava)
    )

    val enum = Schema.createEnum("Aliens", "undocumented", "outer",
      List("Edgar_the_Bug", "Boris_the_Animal", "Laura_Vasquez") asJava)

    val allSchemas = (enum::unions)++simpleSchemas // NULL does not work

    val fields = allSchemas.zipWithIndex map {
      case (s, i) => new Schema.Field("x" + i, s, "Who", null)
    }

    val expected = List(
      AEnum(fields(0), isNullable = false),
      AInt(fields(1), isNullable = true),
      AInt(fields(2), isNullable = true),
      AString(fields(3), isNullable = false),
      AInt(fields(4), isNullable = false),
      ALong(fields(5), isNullable = false),
      AFloat(fields(6), isNullable = false),
      ADouble(fields(7), isNullable = false),
      ABoolean(fields(8), isNullable = false)
    )

    an[IllegalArgumentException] should be thrownBy {
      val nullSchema = Schema.create(Schema.Type.NULL)
      val nullField = new Schema.Field("xxx", null, "Nobody", null)
      AvroField from nullField
    }

    fields.size shouldBe expected.size

    for {
      (field, expected) <- fields zip expected
    } {
      val actual = AvroField from field
      actual shouldBe expected
    }
  }

}
