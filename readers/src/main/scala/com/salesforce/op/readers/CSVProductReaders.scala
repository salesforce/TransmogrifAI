/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.OpParams
import com.salesforce.op.utils.io.csv.CSVOptions
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Encoder, SparkSession}

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * CSV reader for any type that defines an [[Encoder]].
 * Scala case classes and tuples/products included automatically.
 *
 * @param readPath default path to data
 * @param key      function for extracting key from record
 * @param options  CSV options
 * @tparam T
 */
class CSVProductReader[T <: Product : Encoder]
(
  val readPath: Option[String],
  val key: T => String,
  val options: CSVOptions = CSVDefaults.CSVOptions
)(implicit val wtt: WeakTypeTag[T]) extends DataReader[T] with Reader[T] {

  override def read(params: OpParams = new OpParams())(implicit sc: SparkSession): Either[RDD[T], Dataset[T]] = Right {
    val finalPath = getFinalReadPath(params)
    val data: Dataset[T] = sc.read
      .options(options.toSparkCSVOptionsMap)
      .schema(implicitly[Encoder[T]].schema) // without this, every value gets read in as a string
      .csv(finalPath)
      .as[T]
    maybeRepartition(data, params)
  }
}


/**
 * Data Reader for CSV events, where there may be multiple records for a given key. Each csv record
 * will be automatically converted to type T that defines an [[Encoder]].
 * @param readPath default path to data
 * @param key      function for extracting key from record
 * @param options  CSV options
 * @param aggregateParams params for time-based aggregation
 * @tparam T
 */
class AggregateCSVProductReader[T <: Product : Encoder  : WeakTypeTag]
(
  readPath: Option[String],
  key: T => String,
  options: CSVOptions = CSVDefaults.CSVOptions,
  val aggregateParams: AggregateParams[T]
)extends CSVProductReader[T](readPath, key, options) with AggregateDataReader[T]


/**
 * Data Reader for CSV events, when computing conditional probabilities. There may be multiple records for
 * a given key. Each csv record will be automatically converted to type T that defines an [[Encoder]].
 * @param readPath default path to data
 * @param key      function for extracting key from record
 * @param options  CSV options
 * @param conditionalParams params for conditional aggregation
 * @tparam T
 */
class ConditionalCSVProductReader[T <: Product : Encoder : WeakTypeTag]
(
  readPath: Option[String],
  key: T => String,
  options: CSVOptions = CSVDefaults.CSVOptions,
  val conditionalParams: ConditionalParams[T]
)extends CSVProductReader[T](readPath, key, options) with ConditionalDataReader[T]

