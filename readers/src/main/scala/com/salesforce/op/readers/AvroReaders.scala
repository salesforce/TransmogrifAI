/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.OpParams
import com.salesforce.op.utils.io.avro.AvroInOut
import org.apache.avro.generic.GenericRecord
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

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
)(implicit val wtt: WeakTypeTag[T]) extends DataReader[T] with Reader[T] {

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
