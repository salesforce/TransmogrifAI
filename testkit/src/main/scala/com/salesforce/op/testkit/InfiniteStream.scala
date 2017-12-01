/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import language.postfixOps

/**
 * Generates data of given type; it never ends, and it's not resettable.
 * If you want to generate the same data again, instantiate a new generator with the same seed.
 */
trait InfiniteStream[+T] extends Iterator[T] with Serializable {
  self =>
  /**
   * Tell whether there is a next value. The stream is infinite, the answer is Yes.
   * @return true
   */
  override def hasNext: Boolean = true

  /**
   * Transforms this stream to the stream of values of type U, by mapping
   * @param f the function that transforms data
   * @tparam U the type of result of f
   * @return another InfiniteStream, with values produced by applying f
   */
  override def map[U](f: T => U): InfiniteStream[U] = new InfiniteStream[U] {
    def next: U = f(self.next)
  }

  /**
   * Produces a list of n values
   * @param n the number of values
   * @return a list of values
   */
  def limit(n: Int): List[T] = take(n) toList
}
