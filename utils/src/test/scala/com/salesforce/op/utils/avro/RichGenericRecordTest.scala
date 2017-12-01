/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.avro

import com.salesforce.op.test.{TestCommon, TestSparkContext}
import com.salesforce.op.utils.io.avro.AvroInOut
import org.apache.avro.generic.GenericRecord
import org.junit.runner.RunWith
import org.scalatest.{FlatSpec, Matchers}
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class RichGenericRecordTest extends FlatSpec
  with Matchers
  with TestSparkContext
  with TestCommon {

  import com.salesforce.op.utils.avro.RichGenericRecord._

  val dataPath = resourceFile(parent = "../test-data", name = s"PassengerData.avro").getPath
  val passengerData = AvroInOut.read[GenericRecord](dataPath).getOrElse(throw new Exception("Couldn't read data"))
  val firstRow = passengerData.first
  print(firstRow)

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
}
