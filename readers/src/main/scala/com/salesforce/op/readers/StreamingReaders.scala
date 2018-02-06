/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import org.apache.avro.generic.GenericRecord
import org.apache.hadoop.fs.Path

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.WeakTypeTag


/**
 * Just a handy factory for streaming readers
 */
object StreamingReaders {

  /**
   * Simple streaming reader factory
   */
  object Simple {

    /**
     * Creates [[FileStreamingAvroReader]]
     *
     * @param key          function for extracting key from avro record
     * @param filter       Function to filter paths to process
     * @param newFilesOnly Should process only new files and ignore existing files in the directory
     */
    def avro[T <: GenericRecord : ClassTag : WeakTypeTag](
      key: T => String = ReaderKey.randomKey _,
      filter: Path => Boolean = (p: Path) => !p.getName.startsWith("."),
      newFilesOnly: Boolean = false
    ): FileStreamingAvroReader[T] = new FileStreamingAvroReader[T](key, filter, newFilesOnly)

  }

}
