/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.twitter.algebird._

import scala.reflect.runtime.universe._

/**
 * Aggregator that gives the union of Set values
 */
abstract class SetAggregator[A <: MultiPickList](implicit val ttag: WeakTypeTag[A])
  extends MonoidAggregator[Event[A], Set[String], A] with AggregatorDefaults[A] {
  val ftFactory = FeatureTypeFactory[A]()
  val monoid: Monoid[Set[String]] = Monoid.setMonoid[String]
}
case object UnionMultiPickList extends SetAggregator[MultiPickList]

/**
 * Set union semigroup
 */
private[op] class SetSemigroup[T] extends Semigroup[Set[T]] {

  final override def plus(left: Set[T], right: Set[T]): Set[T] =
    if (left.size > right.size) left ++ right else right ++ left

  final override def sumOption(items: TraversableOnce[Set[T]]): Option[Set[T]] =
    if (items.isEmpty) None
    else {
      val mutable = scala.collection.mutable.Set[T]()
      items.foreach { s => mutable ++= s }
      Some(mutable.toSet)
    }
}

