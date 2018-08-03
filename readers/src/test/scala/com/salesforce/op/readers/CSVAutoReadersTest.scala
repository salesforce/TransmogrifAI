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
