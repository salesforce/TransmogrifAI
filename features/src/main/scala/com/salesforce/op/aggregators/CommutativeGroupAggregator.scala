/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.twitter.algebird.{Group, Monoid, MonoidAggregator}

/**
 * Aggregator based on a commutative group
 */
trait CommutativeGroupAggregator[-A, B, +C] extends MonoidAggregator[A, B, C] {
  def group: Group[B]
  override def monoid: Monoid[B] = group
  def plus(x: B, y: B): B = group.plus(x, y)
  def minus(x: B, y: B): B = group.minus(x, y)
}
