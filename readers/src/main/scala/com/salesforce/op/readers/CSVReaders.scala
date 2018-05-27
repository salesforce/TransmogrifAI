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

package com.salesforce.op.readers

import com.salesforce.op.OpParams
import com.salesforce.op.utils.io.csv.{CSVInOut, CSVOptions, CSVToAvro}
import org.apache.avro.generic.GenericRecord
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Data Reader for CSV data. Each CSV record will be automatically converted to an Avro record using the provided
 * schema.
 *
 * @param readPath default path to data
 * @param key      function for extracting key from avro record
 * @param schema   avro schema. Note dateTime fields should be of type Long and will be automatically converted to
 *                 unix timestamps in millis
 * @param options  CSV options
 * @param timeZone timeZone to be used for any dateTime fields
 * @tparam T
 */
class CSVReader[T <: GenericRecord : ClassTag]
(
  val readPath: Option[String],
  val key: T => String,
  val schema: String,
  val options: CSVOptions = CSVDefaults.CSVOptions,
  val timeZone: String = CSVDefaults.TimeZone
)(implicit val wtt: WeakTypeTag[T]) extends DataReader[T] {

  override def read(params: OpParams = new OpParams())
    (implicit spark: SparkSession): Either[RDD[T], Dataset[T]] = Left {
    val finalPath = getFinalReadPath(params)
    val csvData = new CSVInOut(options).readRDD(finalPath)
    val data = CSVToAvro.toAvroTyped[T](csvData = csvData, schemaString = schema, timeZone = timeZone)
    maybeRepartition(data, params)
  }

}

/**
 * Data Reader for event type CSV data, where there may be multiple records for a given key. Each csv record
 * will be automatically converted to an avro record using the provided schema.
 *
 * @param readPath        default path to data
 * @param key             function for extracting key from avro record
 * @param schema          avro schema. Note dateTime fields should be of type Long
 *                        and will be automatically converted to
 *                        unix timestamps in millis
 * @param options         CSV options
 * @param timeZone        timeZone to be used for any dateTime fields
 * @param aggregateParams aggregate params function for extracting timestamp of event
 * @tparam T
 */
class AggregateCSVReader[T <: GenericRecord : ClassTag : WeakTypeTag]
(
  readPath: Option[String],
  key: T => String,
  schema: String,
  options: CSVOptions = CSVDefaults.CSVOptions,
  timeZone: String = CSVDefaults.TimeZone,
  val aggregateParams: AggregateParams[T]
) extends CSVReader[T](readPath = readPath, key = key,
  schema = schema, options = options, timeZone = timeZone) with AggregateDataReader[T]


/**
 * Data Reader for event type CSV data, when computing conditional probabilities. There may be multiple records for
 * a given key. Each csv record will be automatically converted to an avro record using the provided schema.
 *
 * @param readPath default path to data
 * @param key      function for extracting key from avro record
 * @param schema   avro schema. Note dateTime fields should be of type Long and
 *                 will be automatically converted to
 *                 unix timestamps in millis
 * @param options  CSV options
 * @param timeZone timeZone to be used for any dateTime fields
 *
 * @tparam T
 */
class ConditionalCSVReader[T <: GenericRecord : ClassTag : WeakTypeTag]
(
  readPath: Option[String],
  key: T => String,
  schema: String,
  options: CSVOptions = CSVDefaults.CSVOptions,
  timeZone: String = CSVDefaults.TimeZone,
  val conditionalParams: ConditionalParams[T]
) extends CSVReader[T](readPath = readPath, key = key,
  schema = schema, options = options, timeZone = timeZone) with ConditionalDataReader[T]
