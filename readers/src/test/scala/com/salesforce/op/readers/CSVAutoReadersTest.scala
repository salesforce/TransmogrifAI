/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.test.PassengerSparkFixtureTest
import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.collection.JavaConverters._


@RunWith(classOf[JUnitRunner])
class CSVAutoReadersTest extends FlatSpec with PassengerSparkFixtureTest {

  private val expectedSchema = new Schema.Parser().parse(resourceFile(name = "PassengerAuto.avsc"))
  private val allFields = expectedSchema.getFields.asScala.map(_.name())
  private val keyField: String = allFields.head

  Spec[CSVAutoReader[_]] should "read in data correctly and infer schema" in {
    val dataReader = DataReaders.Simple.csvAuto[GenericRecord](
      path = Some(passengerCsvWithHeaderPath),
      key = _.get(keyField).toString
    )
    val data = dataReader.readRDD().collect()
    data.foreach(_ shouldBe a[GenericRecord])
    data.length shouldBe 8

    val inferredSchema = data.head.getSchema
    inferredSchema shouldBe expectedSchema
  }

  it should "read in data correctly and infer schema based with headers provided" in {
    val dataReader = DataReaders.Simple.csvAuto[GenericRecord](
      path = Some(passengerCsvPath),
      key = _.get(keyField).toString,
      headers = allFields
    )
    val data = dataReader.readRDD().collect()
    data.foreach(_ shouldBe a[GenericRecord])
    data.length shouldBe 8

    val inferredSchema = data.head.getSchema
    inferredSchema shouldBe expectedSchema

  }

}
