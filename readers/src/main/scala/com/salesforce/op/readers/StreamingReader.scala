/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.OpParams
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.dstream.DStream

import scala.reflect.ClassTag


trait StreamingReader[T] extends ReaderType[T] with ReaderKey[T] {

  /**
   * Reader class tag
   */
  implicit val ctt: ClassTag[T]

  /**
   * Function which creates a Discretized Stream (DStream) of [T] to read from
   *
   * @param params    parameters used to carry out specialized logic in reader (passed in from workflow)
   * @param streaming spark streaming context
   * @return stream of T to read from
   */
  def stream(params: OpParams)(implicit streaming: StreamingContext): DStream[T]

}
