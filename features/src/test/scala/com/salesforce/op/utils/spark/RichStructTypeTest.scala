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
