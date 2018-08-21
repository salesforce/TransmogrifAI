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

package com.salesforce.op.utils.io.csv

import com.salesforce.op.test.{Passenger, TestSparkContext}
import org.apache.spark.SparkException
import org.apache.spark.rdd.RDD
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class CSVToAvroTest extends FlatSpec with TestSparkContext {
  val avroSchema: String = loadFile(s"$resourceDir/PassengerSchemaModifiedDataTypes.avsc")
  val csvReader: CSVInOut = new CSVInOut(CSVOptions(header = true))
  lazy val csvRDD: RDD[Seq[String]] = csvReader.readRDD(s"$resourceDir/PassengerDataModifiedDataTypes.csv")
  lazy val csvFileRecordCount: Long = csvRDD.count

  Spec(CSVToAvro.getClass) should "convert RDD[Seq[String]] to RDD[GenericRecord]" in {
    val res = CSVToAvro.toAvro(csvRDD, avroSchema)
    res shouldBe a[RDD[_]]
    res.count shouldBe csvFileRecordCount
  }

  it should "convert RDD[Seq[String]] to RDD[T]" in {
    val res = CSVToAvro.toAvroTyped[Passenger](csvRDD, avroSchema)
    res shouldBe a[RDD[_]]
    res.count shouldBe csvFileRecordCount
  }

  it should "throw an error for nested schema" in {
    val invalidAvroSchema = loadFile(s"$resourceDir/PassengerSchemaNestedTypeCSV.avsc")
    val exceptionMsg = "CSV should be a flat file and not have nested records (unsupported column(Sex schemaType=ENUM)"
    val error = intercept[SparkException](CSVToAvro.toAvro(csvRDD, invalidAvroSchema).count())
    error.getCause.getMessage shouldBe exceptionMsg
  }

  it should "throw an error for mis-matching schema fields" in {
    val invalidAvroSchema = loadFile(s"$resourceDir/PassengerSchemaInvalidField.avsc")
    val error = intercept[SparkException](CSVToAvro.toAvro(csvRDD, invalidAvroSchema).count())
    error.getCause.getMessage shouldBe "Mismatch number of fields in csv record and avro schema"
  }

  it should "throw an error for bad data" in {
    val invalidDataRDD = csvReader.readRDD(s"$resourceDir/PassengerDataContentTypeMisMatch.csv")
    val error = intercept[SparkException](CSVToAvro.toAvro(invalidDataRDD, avroSchema).count())
    error.getCause.getMessage shouldBe "Boolean column not actually a boolean. Invalid value: 'fail'"
  }
}
