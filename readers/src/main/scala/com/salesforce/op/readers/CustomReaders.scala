/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.OpParams
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Custom data reader
 *
 * @param key function for extracting key from a record
 */
abstract class CustomReader[T](val key: T => String)(implicit val wtt: WeakTypeTag[T])
  extends DataReader[T] with Reader[T] {

  val readPath: Option[String] = None // dummy, not used in custom readers

  /**
   * User provided function to read the data.
   *
   * @param params parameters used to carry out specialized logic in reader (passed in from workflow)
   * @param spark  spark instance to do the reading and conversion from RDD to Dataframe
   * @return either RDD or Dataset of type T
   */
  def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[T], Dataset[T]]

  override def read(params: OpParams = new OpParams())(implicit spark: SparkSession): Either[RDD[T], Dataset[T]] = {
    readFn(params) match {
      case Left(data) => Left(maybeRepartition(data, params))
      case Right(data) => Right(maybeRepartition(data, params))
    }
  }

}

/**
 * Custom aggregate data reader
 *
 * @param key             function for extracting key from a record
 * @param aggregateParams aggregate params
 * @tparam T
 */
abstract class AggregateCustomReader[T: WeakTypeTag]
(
  key: T => String,
  val aggregateParams: AggregateParams[T]
) extends CustomReader[T](key = key) with AggregateDataReader[T]


/**
 * Custom conditional aggregate data reader
 *
 * @param key               function for extracting key from a record
 * @param conditionalParams conditional probabilities params
 * @tparam T
 */
abstract class ConditionalCustomReader[T: WeakTypeTag]
(
  key: T => String,
  val conditionalParams: ConditionalParams[T]
) extends CustomReader[T](key = key) with ConditionalDataReader[T]
