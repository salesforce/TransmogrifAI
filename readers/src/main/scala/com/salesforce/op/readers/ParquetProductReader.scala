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

import com.salesforce.op.OpParams
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * ParquetReader for any type that defines an [[Encoder]].
 * Scala case classes and tuples/products included automatically.
 *
 * @param readPath default path to data
 * @param key      function for extracting key from record
 * @tparam T
 */
class ParquetProductReader[T <: Product : Encoder]
(
  val readPath: Option[String],
  val key: T => String
)(implicit val wtt: WeakTypeTag[T]) extends DataReader[T] {

  override def read(params: OpParams = new OpParams())(implicit sc: SparkSession): Either[RDD[T], Dataset[T]] = Right {
    val finalPath = getFinalReadPath(params)
    val data: Dataset[T] = sc.read.parquet(finalPath).as[T]
    maybeRepartition(data, params)
  }
}

/**
  * Data Reader for Parquet events, where there may be multiple records for a given key. Each parquet record
  * will be automatically converted to type T that defines an [[Encoder]].
  * @param readPath default path to data
  * @param key      function for extracting key from record
  * @param aggregateParams params for time-based aggregation
  * @tparam T
  */
class AggregateParquetProductReader[T <: Product : Encoder : WeakTypeTag]
(
  readPath: Option[String],
  key: T => String,
  val aggregateParams: AggregateParams[T]
)extends ParquetProductReader[T](readPath, key) with AggregateDataReader[T]

/**
  * Data Reader for Parquet events, when computing conditional probabilities. There may be multiple records for
  * a given key. Each parquet record will be automatically converted to type T that defines an [[Encoder]].
  * @param readPath default path to data
  * @param key      function for extracting key from record
  * @param conditionalParams params for conditional aggregation
  * @tparam T
  */
class ConditionalParquetProductReader[T <: Product : Encoder : WeakTypeTag]
(
  readPath: Option[String],
  key: T => String,
  val conditionalParams: ConditionalParams[T]
)extends ParquetProductReader[T](readPath, key) with ConditionalDataReader[T]
