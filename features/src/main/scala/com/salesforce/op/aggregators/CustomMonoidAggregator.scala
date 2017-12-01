/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types.{FeatureType, FeatureTypeFactory}
import com.twitter.algebird.{Monoid, MonoidAggregator}

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Custom Monoid Aggregator allowing passing a zero value and an associative function to combine values
 *
 * @param zero          zero value
 * @param associativeFn associative function to combine values
 * @tparam O type of feature
 */
case class CustomMonoidAggregator[O <: FeatureType]
(
  zero: O#Value,
  associativeFn: (O#Value, O#Value) => O#Value
)(implicit val ttag: WeakTypeTag[O])
  extends MonoidAggregator[Event[O], O#Value, O] with AggregatorDefaults[O] {
  val ftFactory = FeatureTypeFactory[O]()
  val monoid: Monoid[O#Value] = Monoid.from(zero)(associativeFn)
}
