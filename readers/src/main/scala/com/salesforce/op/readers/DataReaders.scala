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

import org.apache.avro.generic.GenericRecord
import org.apache.spark.sql.Encoder

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.WeakTypeTag
import com.salesforce.op.readers.ReaderKey.randomKey


/**
 * Just a handy factory for data readers
 */
object DataReaders {

  /**
   * Simple data reader factory
   */
  object Simple {

    /**
     * Creates [[CSVReader]]
     */
    def csv[T <: GenericRecord : ClassTag : WeakTypeTag](
      path: Option[String] = None,
      schema: String,
      key: T => String = randomKey _
    ): CSVReader[T] = new CSVReader[T](readPath = path, key = key, schema = schema)

    /**
     * Creates [[CSVAutoReader]]
     */
    def csvAuto[T <: GenericRecord : ClassTag : WeakTypeTag](
      path: Option[String] = None,
      headers: Seq[String] = Seq.empty,
      key: GenericRecord => String = randomKey _
    ): CSVAutoReader[T] = new CSVAutoReader[T](readPath = path, key = key, headers = headers)

    /**
     * Creates [[AvroReader]]
     */
    def avro[T <: GenericRecord : ClassTag : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _
    ): AvroReader[T] = new AvroReader[T](readPath = path, key = key)

    /**
     * Creates a [[CSVProductReader]]
     */
    def csvProduct[T <: Product : Encoder : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _
    ): CSVProductReader[T] = new CSVProductReader[T](readPath = path, key = key)

    /**
     * Creates a [[CSVProductReader]].
     * This method does the same thing as [[csvProduct]], but is called "csvCase" so it is easier to understand.
     */
    def csvCase[T <: Product : Encoder : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _
    ): CSVProductReader[T] = csvProduct(path, key)

    /**
     * Creates a [[ParquetProductReader]]
     */
    def parquetProduct[T <: Product : Encoder : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _
    ): ParquetProductReader[T] = new ParquetProductReader[T](readPath = path, key = key)

    /**
     * Creates a [[ParquetProductReader]].
     * This method does the same thing as [[parquetProduct]], but is called "parquetCase" so it is easier to understand.
     */
    def parquetCase[T <: Product : Encoder : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _
    ): ParquetProductReader[T] = parquetProduct(path, key)

  }

  /**
   * Aggregate data reader factory
   */
  object Aggregate {

    /**
     * Creates [[AggregateCSVReader]]
     */
    def csv[T <: GenericRecord : ClassTag : WeakTypeTag](
      path: Option[String] = None,
      schema: String,
      key: T => String = randomKey _,
      aggregateParams: AggregateParams[T]
    ): AggregateCSVReader[T] = new AggregateCSVReader[T](
      readPath = path, key = key, schema = schema, aggregateParams = aggregateParams
    )

    /**
     * Creates [[AggregateCSVAutoReader]]
     */
    def csvAuto[T <: GenericRecord : ClassTag : WeakTypeTag](
      path: Option[String] = None,
      headers: Seq[String] = Seq.empty,
      key: T => String = randomKey _,
      aggregateParams: AggregateParams[T]
    ): CSVAutoReader[T] = new AggregateCSVAutoReader[T](
      readPath = path, key = key, headers = headers, aggregateParams = aggregateParams
    )

    /**
     * Creates [[AggregateAvroReader]]
     */
    def avro[T <: GenericRecord : ClassTag : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _,
      aggregateParams: AggregateParams[T]
    ): AggregateAvroReader[T] = new AggregateAvroReader[T](
      readPath = path, key = key, aggregateParams = aggregateParams
    )

    /**
     * Creates a [[AggregateCSVProductReader]]
     */
    def csvProduct[T <: Product : Encoder : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _,
      aggregateParams: AggregateParams[T]
    ): AggregateCSVProductReader[T] = new AggregateCSVProductReader[T](
      readPath = path, key = key, aggregateParams = aggregateParams
    )

    /**
     * Creates a [[AggregateCSVProductReader]], but is called csvCase so it's easier to understand
     */
    def csvCase[T <: Product : Encoder : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _,
      aggregateParams: AggregateParams[T]
    ): AggregateCSVProductReader[T] = csvProduct(path, key, aggregateParams)

  }

  /**
   * Conditional data reader factory
   */
  object Conditional {

    /**
     * Creates [[ConditionalCSVReader]]
     */
    def csv[T <: GenericRecord : ClassTag : WeakTypeTag](
      path: Option[String] = None,
      schema: String,
      key: T => String = randomKey _,
      conditionalParams: ConditionalParams[T]
    ): ConditionalCSVReader[T] = new ConditionalCSVReader[T](
      readPath = path, key = key, schema = schema, conditionalParams = conditionalParams
    )

    /**
     * Creates [[ConditionalCSVAutoReader]]
     */
    def csvAuto[T <: GenericRecord : ClassTag : WeakTypeTag](
      path: Option[String] = None,
      headers: Seq[String] = Seq.empty,
      key: T => String = randomKey _,
      conditionalParams: ConditionalParams[T]
    ): ConditionalCSVAutoReader[T] = new ConditionalCSVAutoReader[T](
      readPath = path, key = key, headers = headers, conditionalParams = conditionalParams
    )

    /**
     * Creates [[ConditionalAvroReader]]
     */
    def avro[T <: GenericRecord : ClassTag : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _,
      conditionalParams: ConditionalParams[T]
    ): ConditionalAvroReader[T] = new ConditionalAvroReader[T](
      readPath = path, key = key, conditionalParams = conditionalParams
    )

    /**
     * Creates a [[ConditionalCSVProductReader]]
     */
    def csvProduct[T <: Product : Encoder : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _,
      conditionalParams: ConditionalParams[T]
    ): ConditionalCSVProductReader[T] = new ConditionalCSVProductReader[T](
      readPath = path, key = key, conditionalParams = conditionalParams
    )

    /**
     * Creates a [[ConditionalCSVProductReader]], but is called csvCase so is easier to understand
     */
    def csvCase[T <: Product : Encoder : WeakTypeTag](
      path: Option[String] = None,
      key: T => String = randomKey _,
      conditionalParams: ConditionalParams[T]
    ): ConditionalCSVProductReader[T] = csvProduct(path, key, conditionalParams)

  }

}
