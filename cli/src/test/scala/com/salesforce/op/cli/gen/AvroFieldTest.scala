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
    val simpleSchemas = types map (t => t -> Schema.create(t)) toMap

    val unions = List(
      Schema.createUnion((Schema.Type.NULL::Schema.Type.INT::Nil) map Schema.create asJava),
      Schema.createUnion((Schema.Type.INT::Schema.Type.NULL::Nil) map Schema.create asJava)
    )

    val enum = Schema.createEnum("Aliens", "undocumented", "outer",
      List("Edgar_the_Bug", "Boris_the_Animal", "Laura_Vasquez") asJava)

    val allSchemas = (enum::unions)++simpleSchemas.values.tail // NULL does not work

    val fields = allSchemas.zipWithIndex map {
      case (s, i) => new Schema.Field("x" + i, s, "Who", null)
    }

    val expected = List(
      AEnum(nullable = false, "x0"),
      AInt(nullable = true, "x1"),
      AInt(nullable = true, "x2"),
      AFloat(nullable = false, "x3"),
      ADouble(nullable = false, "x4"),
      ALong(nullable = false, "x5"),
      AString(nullable = false, "x6"),
      AInt(nullable = false, "x7")
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
      val actual = AvroField.from(field)
      actual shouldBe expected
    }
  }

}
