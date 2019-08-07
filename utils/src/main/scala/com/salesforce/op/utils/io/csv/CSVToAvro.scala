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

import java.security.InvalidParameterException

import com.salesforce.op.utils.date.DateTimeUtils
import org.apache.avro.Schema
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.apache.avro.specific.SpecificData
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scala.util.{Success, Try}

object CSVToAvro {

  /**
   * Convert from a buffer of strings to an avro record for converting csvs
   *
   * @param csvData      data that has been read from a csv to an RDD
   * @param schemaString the schema that matches the data
   * @param timeZone     timezone to use in converting time fields, e.g., "GMT+5", "GMT-5", "US/Eastern"
   * @return an RDD of avro generic records
   */
  def toAvro(
    csvData: RDD[Seq[String]], schemaString: String, timeZone: String = DateTimeUtils.DefaultTimeZoneStr
  ): RDD[GenericRecord] = {
    csvData.map(columns => {
      val schema = new Schema.Parser().parse(schemaString)
      toAvroRecord(columns, schema, timeZone)
    })
  }

  /**
   * Convert from a buffer of strings to an avro typed record for converting csvs
   *
   * @param csvData      data that has been read from a csv to an RDD
   * @param schemaString the schema that matches the data
   * @param timeZone     timezone to use in converting time fields, e.g., "GMT+5", "GMT-5", "US/Eastern"
   * @return an RDD of avro typed records
   */
  def toAvroTyped[T <: GenericRecord : ClassTag](
    csvData: RDD[Seq[String]], schemaString: String, timeZone: String = DateTimeUtils.DefaultTimeZoneStr
  ): RDD[T] =
    csvData.map(columns => {
      val schema = new Schema.Parser().parse(schemaString)
      val genericRecord: GenericRecord = toAvroRecord(columns, schema, timeZone)
      SpecificData.get().deepCopy(schema, genericRecord).asInstanceOf[T]
    })

  /**
   * Convert a collection of columns into a generic avro record
   *
   * @param columns  a collection of columns
   * @param schema   avro schema instance
   * @param timeZone timezone to use in converting time fields, e.g., "GMT+5", "GMT-5", "US/Eastern"
   * @return an avro record
   */
  def toAvroRecord(
    columns: Seq[String], schema: Schema, timeZone: String = DateTimeUtils.DefaultTimeZoneStr
  ): GenericRecord = {
    val fields = schema.getFields.asScala
    val genericRecord = new GenericData.Record(schema)
    fields.zipWithIndex.foreach { case (field, index) =>
      val value =
        if (index < columns.size) columns(index)
        else try {
          field.defaultVal().toString
        } catch {
          case e: Exception =>
            throw new InvalidParameterException("Mismatch number of fields in csv record and avro schema")
        }
      genericRecord.put(field.name, transformColumn(value, field.schema, timeZone))
    }
    genericRecord
  }

  // Converts a csv field to properly typed field
  private def transformColumn(
    column: String, colSchema: Schema, timeZone: String
  ): Any = colSchema.getType match {
    // Replacing special chars in a string, so that downstream processing is simpler
    case Schema.Type.STRING => column.replaceAll("\r\n", " ").replaceAll("\"\"", "\"")
    case Schema.Type.INT => column.trim.toInt
    // Some partners give us dates as strings, that we store as longs in avro
    case Schema.Type.LONG => Try(column.trim.toLong).getOrElse(DateTimeUtils.parse(column.trim, timeZone))
    case Schema.Type.BOOLEAN =>
      val colL = column.trim.toLowerCase
      if (Set("1", "true", "yes")(colL)) true
      else if (Set("0", "false", "no")(colL)) false
      else throw new IllegalArgumentException(s"Boolean column not actually a boolean. Invalid value: '$column'")
    case Schema.Type.DOUBLE => column.trim.toDouble
    case Schema.Type.FLOAT => column.trim.toFloat
    case Schema.Type.UNION =>
      if (column == null || column.trim == "" || column.trim.compareToIgnoreCase("null") == 0) null
      else {
        val first :: second :: _ = colSchema.getTypes.asScala.toList
        (first.getType, second.getType) match {
          case (Schema.Type.NULL, _) => transformColumn(column, second, timeZone)
          case (_, Schema.Type.NULL) => transformColumn(column, first, timeZone)
          case _ => throw new IllegalArgumentException("Only support unions for nullable fields in avro schemas")
        }
      }
    case Schema.Type.ENUM => enumBySchema(colSchema, column.trim)

    // should not happen
    case _ => throw new UnsupportedSchemaException(colSchema)
  }

  private def enumBySchema[T <: Enum[T]](schema: Schema, stringValue: String): T = {
    Try(Class.forName(schema.getFullName)) match {
      case Success(klass: Class[T] @unchecked) => Enum.valueOf[T](klass, stringValue)
      case _ => throw new UnsupportedSchemaException(schema)
    }
  }

  class UnsupportedSchemaException(schema: Schema)
    extends Exception(
      "CSV should be a flat file and not have nested records " +
      s"(unsupported column(${schema.getName} schemaType=${schema.getType})"
    )

}
