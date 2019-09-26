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

package com.salesforce.op.cli

import java.io.{File, FileWriter}

import com.salesforce.op.cli.gen.{AvroField, Ops}
import com.salesforce.op.readers.{CSVAutoReader, DataReaders, ReaderKey}
import com.salesforce.op.utils.io.csv.CSVOptions
import com.salesforce.op.utils.kryo.OpKryoRegistrator
import org.apache.avro.Schema
import org.apache.avro.generic.GenericRecord
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConverters._

/**
 * A variety of functionalities for pulling data schema.
 * One way is from Avro (as usual), another is automatic, by applying AutoReaders
 */
sealed trait SchemaSource {
  val name: String
  val fullName: String
  def dataSchema: Schema
  lazy val fields: List[AvroField] = dataSchema.getFields.asScala.toList.map(AvroField.from)

  def theReader: String

  def schemaFile: File

  protected def tentativeResponseField(name: String): AvroField =
    exactlyOne(fields, name.toLowerCase, fieldPurpose = "Response")

  def responseField(name: String): AvroField

  def idField(name: String): AvroField = exactlyOne(fields, name.toLowerCase, fieldPurpose = "Id")

  /**
   * Get _exactly one_ [[AvroField]] by name from a list of the schema's fields. If exactly one isn't found, exit
   * with an error. Not sensitive to case.
   *
   * @param l         A list of [[AvroField]]s from the schema's fields
   * @param fieldName The name of the field we're searching for
   * @param fieldPurpose The kind of the field we're trying to find exactly one of, typically "response" or "id"
   * @return The [[AvroField]] representing the field we passed in
   */
  private def exactlyOne(l: List[AvroField], fieldName: String, fieldPurpose: String): AvroField = {
    l.filter { _.name.toLowerCase == fieldName } match {
      case Nil => Ops.oops(s"$fieldPurpose field '$fieldName' not found (ignoring case)")
      case schema :: Nil => schema
      case _ => Ops.oops(s"$fieldPurpose field '$fieldName' is defined more than once in the schema (ignoring case)")
    }
  }
}

case class AutomaticSchema(recordClassName: String)(dataFile: File) extends SchemaSource {
  val name: String = recordClassName
  private val defaultName = "AutoInferredRecord"
  lazy val fullName: String = dataSchema.getFullName.replace(defaultName, name)
  val theReader = "ReaderWithHeaders"

  lazy val conf: SparkConf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("op-cli-schema")
    .set("spark.serializer", classOf[org.apache.spark.serializer.KryoSerializer].getName)
    .set("spark.kryo.registrator", classOf[OpKryoRegistrator].getName)
    .set("spark.ui.enabled", false.toString)

  implicit lazy val spark: SparkSession = SparkSession.builder.config(conf).getOrCreate()
  implicit lazy val sc: SparkContext = spark.sparkContext

  lazy val dataSchema: Schema = {
    println(
      s"""Launching spark to read data from the file $dataFile to deduce data schema.
      |This may take a while, depending on the input file size.""".stripMargin)
    val dataReader1 = DataReaders.Simple.csvAuto[GenericRecord](path = Option(dataFile.getAbsolutePath))

    val dataReader: CSVAutoReader[GenericRecord] =
      new CSVAutoReader[GenericRecord](
        readPath = Option(dataFile.getAbsolutePath),
        key = ReaderKey.randomKey _,
        headers = Seq.empty,
        options = new CSVOptions(format = "org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
      )

    val hadoopConfig = sc.hadoopConfiguration
    hadoopConfig.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
    hadoopConfig.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)

    val schema = try {
      dataReader.readRDD().first.getSchema
    } catch {
      case spe: org.apache.avro.SchemaParseException
        if spe.getMessage contains "Illegal initial" =>
          throw BadSchema(
            s"Bad data file, it should contain column headers, and $dataFile does not",
            spe)
      case x: Exception => throw BadSchema("Failed to build Schema", x)
    }
    schema
  }

  override def responseField(name: String): AvroField = {
    val field = tentativeResponseField(name)
    if (field.isNullable) {
      val orgSchemaField = field.schemaField
      AvroField.typeOfNullable(orgSchemaField.schema) match {
        case None => Ops.oops(s"Could not provide non-nullable response filed '$name', field was $orgSchemaField")
        case Some(actualType) =>
          val newSchema = Schema.create(actualType)
          val schemaField =
            new Schema.Field(field.name, newSchema, "auto-generated", orgSchemaField.defaultValue)
          AvroField.from(schemaField)
      }
    } else field
  }

  override def schemaFile: File = {
    val goodName = recordClassName.capitalize
    val file = File.createTempFile(goodName, ".avsc")
    val out = new FileWriter(file)
    val s = dataSchema.toString(true).replace(defaultName, goodName)
    out.write(s)
    out.close()
    file
  }
}

case class AvroSchemaFromFile(schemaFile: File) extends SchemaSource {
  val dataSchema: Schema = {
    val parser = new Schema.Parser
    parser.parse(schemaFile)
  }
  if (dataSchema.getType != Schema.Type.RECORD) {
    Ops.oops(s"Schema '${dataSchema.getFullName}' must be a record schema, check out $schemaFile")
  }

  val name: String = dataSchema.getName
  lazy val fullName: String = dataSchema.getFullName
  val theReader = "ReaderWithNoHeaders"

  def responseField(name: String): AvroField = {
    val field = tentativeResponseField(name)
    if (field.isNullable) {
      Ops.oops(s"Response field '$field' cannot be nullable in $dataSchema")
    }
    field
  }

}

case class BadSchema(msg: String, source: Exception) extends Exception(msg, source)
