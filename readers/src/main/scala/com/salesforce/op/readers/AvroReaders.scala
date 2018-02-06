/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.OpParams
import com.salesforce.op.utils.io.avro.AvroInOut
import org.apache.avro.generic.GenericRecord
import org.apache.avro.mapred.AvroKey
import org.apache.avro.mapreduce.AvroKeyInputFormat
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.NullWritable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.dstream.DStream

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Data reader for avro data.
 *
 * @param readPath default path to data
 * @param key      function for extracting key from avro record
 * @tparam T
 */
class AvroReader[T <: GenericRecord : ClassTag]
(
  val readPath: Option[String],
  val key: T => String
)(implicit val wtt: WeakTypeTag[T]) extends DataReader[T] {

  override def read(params: OpParams = new OpParams())
    (implicit spark: SparkSession): Either[RDD[T], Dataset[T]] = Left {
    val finalPath = getFinalReadPath(params)
    val data = AvroInOut.readPathSeq[T](finalPath, withCount = true)
    maybeRepartition(data, params)
  }

}

/**
 * Data reader for avro events where there might be multiple records for a given key.
 *
 * @param readPath        default path to data
 * @param key             function for extracting key from avro record
 * @param aggregateParams aggregate params
 * @tparam T
 */
class AggregateAvroReader[T <: GenericRecord : ClassTag : WeakTypeTag]
(
  readPath: Option[String],
  key: T => String,
  val aggregateParams: AggregateParams[T]
) extends AvroReader[T](readPath, key) with AggregateDataReader[T]


/**
 * Data reader for avro events when computing conditional probabilities.
 *
 * @param readPath          default path to data
 * @param key               function for extracting key from avro record
 * @param conditionalParams conditional probabilities params
 * @tparam T
 */
class ConditionalAvroReader[T <: GenericRecord : ClassTag : WeakTypeTag]
(
  readPath: Option[String],
  key: T => String,
  val conditionalParams: ConditionalParams[T]
) extends AvroReader[T](readPath, key) with ConditionalDataReader[T]


/**
 * Simple avro streaming reader that monitors a Hadoop-compatible filesystem for new files.
 *
 * @param key          function for extracting key from avro record
 * @param filter       Function to filter paths to process
 * @param newFilesOnly Should process only new files and ignore existing files in the directory
 * @tparam T
 */
class FileStreamingAvroReader[T <: GenericRecord]
(
  val key: T => String,
  val filter: Path => Boolean,
  val newFilesOnly: Boolean
)(implicit val ctt: ClassTag[T], val wtt: WeakTypeTag[T]) extends StreamingReader[T] {

  /**
   * Function which creates an stream of T to read from
   *
   * @param params    parameters used to carry out specialized logic in reader (passed in from workflow)
   * @param streaming spark streaming context
   * @return stream of T to read from
   */
  def stream(params: OpParams)(implicit streaming: StreamingContext): DStream[T] = {
    val readerParams = getReaderParams(params)
    require(readerParams.flatMap(_.path).isDefined, "The path is not set")
    val readPath = readerParams.flatMap(_.path).get

    streaming.fileStream[AvroKey[T], NullWritable, AvroKeyInputFormat[T]](
      directory = readPath, filter = filter, newFilesOnly = newFilesOnly
    ).map(_._1.datum())
  }

}
