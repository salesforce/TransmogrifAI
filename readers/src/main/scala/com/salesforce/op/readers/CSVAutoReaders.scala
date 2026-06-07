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

import org.apache.spark.sql.avro.SchemaConverters
import com.salesforce.op.OpParams
import com.salesforce.op.utils.io.csv.{CSVInOut, CSVOptions, CSVToAvro}
import org.apache.avro.generic.GenericRecord
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.csv.CSVSchemaUtils
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Data Reader for CSV data that automatically infers the schema from the CSV data and converts to T <: GenericRecord.
 * The schema is inferred either using the provided headers params, otherwise the first row is assumed as a headers line
 *
 * @param readPath        default path to data
 * @param key             function for extracting key from avro record
 * @param headers         header of the CSV file as array, otherwise the first row is assumed as a headers line
 * @param options         CSV options
 * @param timeZone        timeZone to be used for any dateTime fields
 * @param recordNamespace result record namespace
 * @param recordName      result record name
 * @tparam T
 */
class CSVAutoReader[T <: GenericRecord : ClassTag]
(
  val readPath: Option[String],
  val key: T => String,
  val headers: Seq[String] = Seq.empty,
  val options: CSVOptions = CSVDefaults.CSVOptions,
  val timeZone: String = CSVDefaults.TimeZone,
  val recordNamespace: String = CSVDefaults.RecordNamespace,
  val recordName: String = CSVDefaults.RecordName
)(implicit val wtt: WeakTypeTag[T]) extends DataReader[T] {

  override def read(params: OpParams = new OpParams())
    (implicit spark: SparkSession): Either[RDD[T], Dataset[T]] = Left {
    val finalPath = getFinalReadPath(params)
    val csvData = new CSVInOut(options).readRDD(finalPath)
    val hdrs = if (headers.nonEmpty) headers else csvData.first()
    val hdrsSet = hdrs.toSet
    val data = csvData.filter(_.exists(!hdrsSet.contains(_)))

    val columnPrunning = spark.sessionState.conf.csvColumnPruning
    val inferredSchema = CSVSchemaUtils.infer(data.map(_.toArray), hdrs, options, columnPrunning)
    val schema = SchemaConverters.toAvroType(
      inferredSchema, nullable = false, recordName = recordName, nameSpace = recordNamespace
    )
    val avroData: RDD[T] = CSVToAvro.toAvroTyped[T](data, schema.toString, timeZone)
    maybeRepartition(avroData, params)
  }

}

/**
 * Data Reader for event type CSV data, where there may be multiple records for a given key. Each csv record
 * will be automatically converted to an avro record by inferring a schema.
 *
 * @param readPath        default path to data
 * @param key             function for extracting key from avro record
 * @param headers         header of the CSV file as array, otherwise the first row is assumed as a headers line
 * @param options         CSV options
 * @param timeZone        timeZone to be used for any dateTime fields
 * @param recordNamespace result record namespace
 * @param recordName      result record name
 * @param aggregateParams aggregate params function for extracting timestamp of event
 * @tparam T
 */
class AggregateCSVAutoReader[T <: GenericRecord : ClassTag]
(
  readPath: Option[String],
  key: T => String,
  headers: Seq[String] = Seq.empty,
  options: CSVOptions = CSVDefaults.CSVOptions,
  timeZone: String = CSVDefaults.TimeZone,
  recordNamespace: String = CSVDefaults.RecordNamespace,
  recordName: String = CSVDefaults.RecordName,
  val aggregateParams: AggregateParams[T]
) extends CSVAutoReader(readPath = readPath, key = key,
  headers = headers, options = options, timeZone = timeZone,
  recordNamespace = recordNamespace, recordName = recordName) with AggregateDataReader[T]


/**
 * Data Reader for event type CSV data (with schema inference), when computing conditional probabilities.
 * There may be multiple records for a given key.
 * Each csv record will be automatically converted to an avro record with an inferred schema.
 *
 * @param readPath        default path to data
 * @param key             function for extracting key from avro record
 * @param headers         header of the CSV file as array, otherwise the first row is assumed as a headers line
 * @param options         CSV options
 * @param timeZone        timeZone to be used for any dateTime fields
 * @param recordNamespace result record namespace
 * @param recordName      result record name
 * @tparam T
 */
class ConditionalCSVAutoReader[T <: GenericRecord : ClassTag]
(
  readPath: Option[String],
  key: T => String,
  headers: Seq[String] = Seq.empty,
  options: CSVOptions = CSVDefaults.CSVOptions,
  timeZone: String = CSVDefaults.TimeZone,
  recordNamespace: String = CSVDefaults.RecordNamespace,
  recordName: String = CSVDefaults.RecordName,
  val conditionalParams: ConditionalParams[T]
) extends CSVAutoReader(readPath = readPath, key = key,
  headers = headers, options = options, timeZone = timeZone,
  recordNamespace = recordNamespace, recordName = recordName) with ConditionalDataReader[T]
