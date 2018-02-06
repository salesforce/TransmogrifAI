/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
