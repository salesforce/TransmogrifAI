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

package com.salesforce.op.utils.avro

import com.salesforce.op.test.{TestCommon, TestSparkContext}
import com.salesforce.op.utils.io.avro.AvroInOut
import org.apache.avro.generic.GenericRecord
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class RichGenericRecordTest extends FlatSpec
  with Matchers
  with TestSparkContext
  with TestCommon {

  import com.salesforce.op.utils.avro.RichGenericRecord._

  val dataPath = resourceFile(parent = "../test-data", name = s"PassengerData.avro").getPath
  val passengerData = AvroInOut.read[GenericRecord](dataPath).getOrElse(throw new Exception("Couldn't read data"))
  val firstRow = passengerData.first

  Spec[RichGenericRecord] should "get value of Int" in {
    val id = firstRow.getValue[Int]("passengerId")
    id shouldBe Some(1)
  }

  it should "get value of Double" in {
    val survived = firstRow.getValue[Double]("survived")
    survived shouldBe Some(0.0)
  }

  it should "get value of Long" in {
    val height = firstRow.getValue[Long]("height")
    height shouldBe Some(168L)
  }

  it should "get value of String" in {
    val gender = firstRow.getValue[String]("gender")
    gender shouldBe Some("Female")
  }

  it should "get value of Char" in {
    val gender = firstRow.getValue[Char]("gender")
    gender shouldBe Some("Female")
  }

  it should "get value of Float" in {
    val age = firstRow.getValue[Float]("age")
    age shouldBe Some(32.0)
  }

  it should "get value of Short" in {
    val weight = firstRow.getValue[Short]("weight")
    weight shouldBe Some(67)
  }

  it should "throw error for invalid field" in {
    val error = intercept[IllegalArgumentException](firstRow.getValue[Short]("invalidField"))
    error.getMessage shouldBe "requirement failed: invalidField is not found in Avro schema!"
  }
}
