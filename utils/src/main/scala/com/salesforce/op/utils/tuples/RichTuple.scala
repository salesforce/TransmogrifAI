/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.tuples


/**
 * Enrichment functions for tuples
 */
case object RichTuple {

  implicit class RichOptionTuple2[T](val v: (Option[T], Option[T])) extends AnyVal {

    /**
     * Apply specified fun on tuple values if both sides are non empty
     *
     * @param fun function to apply on tuple values if both sides are non empty
     * @return option value
     */
    def map(fun: (T, T) => T): Option[T] =
      if (v._1.isEmpty) v._2 else if (v._2.isEmpty) v._1 else Option(fun(v._1.get, v._2.get))

  }

}
