/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import com.salesforce.op.test.TestSparkContext
import org.apache.spark.sql.functions._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.language.postfixOps


@RunWith(classOf[JUnitRunner])
class RichStructTypeTest extends FlatSpec with TestSparkContext {

  import com.salesforce.op.utils.spark.RichStructType._

  case class Human
  (
    name: String,
    age: Double,
    height: Double,
    heightIsNull: Double,
    isBlueEyed: Double,
    gender: Double,
    testFeatNegCor: Double
  )

  // scalastyle:off
  val humans = Seq(
    Human("alex",     32,  5.0,  0,  1,  1,  0),
    Human("alice",    32,  4.0,  1,  0,  0,  1),
    Human("bob",      32,  6.0,  1,  1,  1,  0),
    Human("charles",  32,  5.5,  0,  1,  1,  0),
    Human("diana",    32,  5.4,  1,  0,  0,  1),
    Human("max",      32,  5.4,  1,  0,  0,  1)
  )
  // scalastyle:on

  val humansDF = spark.createDataFrame(humans).select(col("*"), col("name").as("(name)_blarg_123"))
  val schema = humansDF.schema

  Spec[RichStructType] should "find schema fields by name (case insensitive)" in {
    schema.findFields("name").map(_.name) shouldBe Seq("name", "(name)_blarg_123")
    schema.findFields("blArg").map(_.name) shouldBe Seq("(name)_blarg_123")
  }

  it should "find schema fields by name (case sensitive)" in {
    schema.findFields("Name", ignoreCase = false) shouldBe Seq.empty
    schema.findFields("aGe", ignoreCase = false) shouldBe Seq.empty
    schema.findFields("age", ignoreCase = false).map(_.name) shouldBe Seq("age")
  }

  it should "fail on duplication" in {
    the[IllegalArgumentException] thrownBy schema.findField("a")
  }

  it should "throw an error if no such name" in {
    the[IllegalArgumentException] thrownBy schema.findField("???")
  }

}
