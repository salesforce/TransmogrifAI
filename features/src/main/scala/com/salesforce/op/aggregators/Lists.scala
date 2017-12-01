/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.twitter.algebird._

import scala.reflect.runtime.universe._

/**
 * Aggregator that gives concatenation of the lists
 */
abstract class ConcatList[V, T <: OPList[V]](implicit val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Seq[V], T]
    with AggregatorDefaults[T] {
  val ftFactory: FeatureTypeFactory[T] = FeatureTypeFactory[T]()
  val monoid: Monoid[Seq[V]] = Monoid.seqMonoid[V]

}
case object ConcatTextList extends ConcatList[String, TextList]
case object ConcatDateList extends ConcatList[Long, DateList]
case object ConcatDateTimeList extends ConcatList[Long, DateTimeList]


/**
 * Aggregator that gives min/max item over a collection of lists
 */
abstract class MinMaxList[N, T <: OPList[N]]
(
  isMin: Boolean
)(implicit val ord: Ordering[N], ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Seq[N], T] {
  val ftFactory: FeatureTypeFactory[T] = FeatureTypeFactory[T]()

  def prepare(input: Event[T]): Seq[N] = prepareFn(input.value.value)
  def present(reduction: Seq[N]): T = ftFactory.newInstance(prepareFn(reduction))
  private def prepareFn(v: Seq[N]): Seq[N] = if (v.isEmpty) Seq.empty else Seq(v.reduce(ordFn))

  val monoid: Monoid[Seq[N]] = Monoid.seqMonoid[N]
  val ordFn: (N, N) => N = if (isMin) ord.min _ else ord.max _

}
case object MaxDateList extends MinMaxList[Long, DateList](isMin = false)
case object MaxDateTimeList extends MinMaxList[Long, DateTimeList](isMin = false)
case object MinDateList extends MinMaxList[Long, DateList](isMin = true)
case object MinDateTimeList extends MinMaxList[Long, DateTimeList](isMin = true)
