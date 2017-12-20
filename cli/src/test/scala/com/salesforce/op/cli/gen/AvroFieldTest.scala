/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import language.postfixOps
import collection.JavaConverters._

import com.salesforce.op.test.TestCommon
import org.apache.avro.Schema
import org.scalatest.{Assertions, FlatSpec}
import AvroField._

/**
 * Test for AvroField methods.
 */
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
