/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.salesforce.op.utils.tuples.RichTuple._
import com.twitter.algebird._

import scala.reflect.runtime.universe._


/**
 * Aggregator that gives the sum of the numeric values
 */
abstract class SumNumeric[N: Semigroup, T <: OPNumeric[N]](implicit val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Option[N], T] with AggregatorDefaults[T] {
  val ftFactory = FeatureTypeFactory[T]()
  val monoid = Monoid.optionMonoid[N]
}
case object SumReal extends SumNumeric[Double, Real]
case object SumRealNN extends SumNumeric[Double, RealNN]
case object SumCurrency extends SumNumeric[Double, Currency]
case object SumIntegral extends SumNumeric[Long, Integral]

/**
 * Aggregator that gives the min or max of the numeric values
 */
abstract class MinMaxNumeric[N, T <: OPNumeric[N]]
(
  isMin: Boolean
)(implicit val ord: Ordering[N], val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Option[N], T] with AggregatorDefaults[T] {
  val ordFn = if (isMin) ord.min _ else ord.max _
  val ftFactory = FeatureTypeFactory[T]()
  val monoid = Monoid.from[Option[N]](None)((l, r) => (l -> r).map(ordFn))
}
case object MaxReal extends MinMaxNumeric[Double, Real](isMin = false)
case object MaxCurrency extends MinMaxNumeric[Double, Currency](isMin = false)
case object MaxIntegral extends MinMaxNumeric[Long, Integral](isMin = false)
case object MaxDate extends MinMaxNumeric[Long, Date](isMin = false)
case object MaxDateTime extends MinMaxNumeric[Long, DateTime](isMin = false)
case object MinReal extends MinMaxNumeric[Double, Real](isMin = true)
case object MinCurrency extends MinMaxNumeric[Double, Currency](isMin = true)
case object MinIntegral extends MinMaxNumeric[Long, Integral](isMin = true)
case object MinDate extends MinMaxNumeric[Long, Date](isMin = true)
case object MinDateTime extends MinMaxNumeric[Long, DateTime](isMin = true)


/**
 * Aggregator that gives the mean of the real values
 */
abstract class MeanDouble[A <: OPNumeric[Double]](presentFn: Option[Double] => A)
  extends MonoidAggregator[Event[A], Option[(Double, Int)], A] {
  def prepare(input: Event[A]): Option[(Double, Int)] = input.value.value.map((_, 1))
  def present(reduction: Option[(Double, Int)]): A = presentFn(reduction.map {
    case (sum, count) if count != 0 => sum / count
    case _ => 0.0
  })
  val monoid: Monoid[Option[(Double, Int)]] = Monoid.optionMonoid[(Double, Int)]
}
case object MeanReal extends MeanDouble[Real](new Real(_))
case object MeanCurrency extends MeanDouble[Currency](new Currency(_))
case object MeanPercent extends MeanDouble[Percent](new Percent(_)) with PercentPrepare {
  override def prepare(input: Event[Percent]): Option[(Double, Int)] = input.value.value.map(prepareFn(_) -> 1)
}

/**
 * Aggregator that gives the logical operation of the binary values
 */
abstract class LogicalOp
  extends MonoidAggregator[Event[Binary], Option[Boolean], Binary] with AggregatorDefaults[Binary] {
  implicit val ttag = weakTypeTag[Binary]
  val ftFactory = FeatureTypeFactory[Binary]()
  val monoid: Monoid[Option[Boolean]]
}
case object LogicalOr extends LogicalOp with LogicalOrMonoid
case object LogicalXor extends LogicalOp with LogicalXorMonoid
case object LogicalAnd extends LogicalOp with LogicalAndMonoid


private[op] trait PercentPrepare {
  def prepareFn(percent: Double): Double = percent match {
    case p if p < 0.0 => 0.0
    case p if p > 1.0 => 1.0
    case p => p
  }
}

private[op] trait LogicalOrMonoid {
  val monoid: Monoid[Option[Boolean]] = new Monoid[Option[Boolean]] {
    val zero: Option[Boolean] = None
    def plus(l: Option[Boolean], r: Option[Boolean]): Option[Boolean] = (l -> r).map(_ || _)
  }
}

private[op] trait LogicalXorMonoid {
  val monoid: Monoid[Option[Boolean]] = new Monoid[Option[Boolean]] {
    val zero: Option[Boolean] = None
    def plus(l: Option[Boolean], r: Option[Boolean]): Option[Boolean] = (l -> r).map(_ ^ _)
  }
}

private[op] trait LogicalAndMonoid {
  val monoid: Monoid[Option[Boolean]] = new Monoid[Option[Boolean]] {
    val zero: Option[Boolean] = None
    def plus(l: Option[Boolean], r: Option[Boolean]): Option[Boolean] = (l -> r).map(_ && _)
  }
}
