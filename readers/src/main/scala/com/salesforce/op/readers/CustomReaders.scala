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
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Custom data reader
 *
 * @param key function for extracting key from a record
 */
abstract class CustomReader[T](val key: T => String)(implicit val wtt: WeakTypeTag[T]) extends DataReader[T] {

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
