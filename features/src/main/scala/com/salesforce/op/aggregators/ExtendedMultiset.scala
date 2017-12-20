/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.twitter.algebird._

/**
 * Multiset with possible negative counters (so what)
 *
 * Multiset - is a generalization of the concept of a set that,
 * unlike a set, allows multiple instances of the multiset's elements.
 * For example, {a, a, b} and {a, b} are different multisets although they are the same set.
 * However, order does not matter, so {a, a, b} and {a, b, a} are the same multiset.
 */
trait ExtendedMultiset extends MapMonoid[String, Long] with Group[Map[String, Long]] {
  override def minus(x: Map[String, Long], y: Map[String, Long]): Map[String, Long] = {
    val keys = x.keySet ++ y.keySet
    val kvPairs = keys map (k => k -> (x.getOrElse(k, 0L) - y.getOrElse(k, 0L))) filter (_._2 != 0L)
    kvPairs.toMap
  }
}

object ExtendedMultiset extends ExtendedMultiset
