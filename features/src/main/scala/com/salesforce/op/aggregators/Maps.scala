/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.salesforce.op.utils.text.TextUtils
import com.twitter.algebird._

import scala.reflect.runtime.universe._


/**
 * Aggregator that gives the union of numeric data
 */
abstract class UnionSumNumericMap[N: Numeric, T <: OPMap[N]](implicit val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Map[String, N], T]
    with AggregatorDefaults[T] {
  val num: Numeric[N] = implicitly[Numeric[N]]
  val numericSemigroup: Semigroup[N] = Semigroup.from[N](num.plus)
  val ftFactory: FeatureTypeFactory[T] = FeatureTypeFactory[T]()
  val monoid: Monoid[Map[String, N]] = Monoid.mapMonoid[String, N](numericSemigroup)
}
case object UnionCurrencyMap extends UnionSumNumericMap[Double, CurrencyMap]
case object UnionRealMap extends UnionSumNumericMap[Double, RealMap]
case object UnionIntegralMap extends UnionSumNumericMap[Long, IntegralMap]


/**
 * Natural map monoid lifting averaging operator
 */
abstract class UnionMeanDoubleMap[T <: OPMap[Double]](implicit val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Map[String, (Double, Int)], T] {
  val ftFactory: FeatureTypeFactory[T] = FeatureTypeFactory[T]()
  def prepare(input: Event[T]): Map[String, (Double, Int)] = input.value.value.map { case (k, v) => k -> (v, 1) }
  def present(reduction: Map[String, (Double, Int)]): T = ftFactory.newInstance(reduction.map {
    case (k, (sum, count)) if count != 0 => k -> (sum / count)
    case (k, _) => k -> 0.0
  })
  val monoid: Monoid[Map[String, (Double, Int)]] = Monoid.mapMonoid[String, (Double, Int)]
}
case object UnionMeanCurrencyMap extends UnionMeanDoubleMap[CurrencyMap]
case object UnionMeanRealMap extends UnionMeanDoubleMap[RealMap]
case object UnionMeanPercentMap extends UnionMeanDoubleMap[PercentMap] with PercentPrepare {
  override def prepare(input: Event[PercentMap]): Map[String, (Double, Int)] =
    input.value.value.map { case (k, p) => k -> (prepareFn(p), 1) }
}

case object UnionGeolocationMidpointMap
  extends MonoidAggregator[Event[GeolocationMap], Map[String, Array[Double]], GeolocationMap]
    with GeolocationFunctions {
  /**
   * Prepare method to be used in the MonoidAggregator for GeolocationMap objects
   *
   * @param input Event-wrapped GeolocationMap object
   * @return Map of key -> Array of (x,y,z,acc,count) to be used during aggregation
   */
  def prepare(input: Event[GeolocationMap]): Map[String, Array[Double]] =
    input.value.value.map { case (k, v) => k -> prepare(Geolocation(v)) }

  /**
   * Present method to be used in the MonoidAggregator for GeolocationMap objects
   *
   * @param reduction Map of key -> Array of (x,y,z,acc,count) to be used during aggregation
   * @return Map of key -> Geolocation object corresponding to aggregated x,y,z coordinates
   */
  def present(reduction: Map[String, Array[Double]]): GeolocationMap = GeolocationMap(
    reduction.map { case (k, v) => k -> present(v).value }
  )

  val monoid: Monoid[Map[String, Array[Double]]] = Monoid.mapMonoid[String, Array[Double]](GeolocationMidpoint.monoid)
}


/**
 * Natural map monoid for Map[String, T <: OPNumeric[N]] where N is totally ordered,
 * by lifting max(_, _) monoid operation to a monoid operator on map.
 */
abstract class UnionMinMaxNumericMap[N, T <: OPMap[N]]
(
  isMin: Boolean
)(implicit val ord: Ordering[N], val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Map[String, N], T]
    with AggregatorDefaults[T] {
  val ordFn: (N, N) => N = if (isMin) ord.min _ else ord.max _
  val orderMapSemigroup: Semigroup[N] = Semigroup.from[N](ordFn)
  val ftFactory: FeatureTypeFactory[T] = FeatureTypeFactory[T]()
  val monoid: Monoid[Map[String, N]] = Monoid.mapMonoid[String, N](orderMapSemigroup)
}
case object UnionMaxRealMap extends UnionMinMaxNumericMap[Double, RealMap](isMin = false)
case object UnionMaxCurrencyMap extends UnionMinMaxNumericMap[Double, CurrencyMap](isMin = false)
case object UnionMaxIntegralMap extends UnionMinMaxNumericMap[Long, IntegralMap](isMin = false)
case object UnionMaxDateMap extends UnionMinMaxNumericMap[Long, DateMap](isMin = false)
case object UnionMaxDateTimeMap extends UnionMinMaxNumericMap[Long, DateTimeMap](isMin = false)
case object UnionMinRealMap extends UnionMinMaxNumericMap[Double, RealMap](isMin = true)
case object UnionMinCurrencyMap extends UnionMinMaxNumericMap[Double, CurrencyMap](isMin = true)
case object UnionMinIntegralMap extends UnionMinMaxNumericMap[Long, IntegralMap](isMin = true)
case object UnionMinDateMap extends UnionMinMaxNumericMap[Long, DateMap](isMin = true)
case object UnionMinDateTimeMap extends UnionMinMaxNumericMap[Long, DateTimeMap](isMin = true)


/**
 * Aggregator that gives the union of text map data, concatenating the values with a separator on matching keys
 */
abstract class UnionConcatTextMap[T <: OPMap[String]](val separator: String)(implicit val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Map[String, String], T]
    with AggregatorDefaults[T] {
  val stringSepSemigroup: Semigroup[String] = Semigroup.from[String](TextUtils.concat(_, _, separator = separator))
  val ftFactory: FeatureTypeFactory[T] = FeatureTypeFactory[T]()
  val monoid: Monoid[Map[String, String]] = Monoid.mapMonoid[String, String](stringSepSemigroup)
}
case object UnionConcatBase64Map extends UnionConcatTextMap[Base64Map](separator = ",")
case object UnionConcatComboBoxMap extends UnionConcatTextMap[ComboBoxMap](separator = ",")
case object UnionConcatEmailMap extends UnionConcatTextMap[EmailMap](separator = ",")
case object UnionConcatIDMap extends UnionConcatTextMap[IDMap](separator = ",")
case object UnionConcatPhoneMap extends UnionConcatTextMap[PhoneMap](separator = ",")
case object UnionConcatPickListMap extends UnionConcatTextMap[PickListMap](separator = ",")
case object UnionConcatTextMap extends UnionConcatTextMap[TextMap](separator = " ")
case object UnionConcatTextAreaMap extends UnionConcatTextMap[TextAreaMap](separator = " ")
case object UnionConcatURLMap extends UnionConcatTextMap[URLMap](separator = ",")
case object UnionConcatCountryMap extends UnionConcatTextMap[CountryMap](separator = ",")
case object UnionConcatStateMap extends UnionConcatTextMap[StateMap](separator = ",")
case object UnionConcatCityMap extends UnionConcatTextMap[CityMap](separator = ",")
case object UnionConcatPostalCodeMap extends UnionConcatTextMap[PostalCodeMap](separator = ",")
case object UnionConcatStreetMap extends UnionConcatTextMap[StreetMap](separator = ",")


/**
 * Aggregator that gives the union of binary map data
 */
case object UnionBinaryMap
  extends MonoidAggregator[Event[BinaryMap], Map[String, Boolean], BinaryMap]
    with AggregatorDefaults[BinaryMap] {
  implicit val ttag = weakTypeTag[BinaryMap]
  val logicalOrSemigroup = Semigroup.from[Boolean](_ || _)
  val ftFactory: FeatureTypeFactory[BinaryMap] = FeatureTypeFactory[BinaryMap]()
  val monoid: Monoid[Map[String, Boolean]] = Monoid.mapMonoid[String, Boolean](logicalOrSemigroup)
}

/**
 * Aggregator that gives the union of set map data
 */
abstract class UnionSetMap[T <: OPMap[Set[String]]](implicit val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Map[String, Set[String]], T]
    with AggregatorDefaults[T] {
  val setSemigroup = new SetSemigroup[String]
  val ftFactory: FeatureTypeFactory[T] = FeatureTypeFactory[T]()
  val monoid = Monoid.mapMonoid[String, Set[String]](setSemigroup)
}
case object UnionMultiPickListMap extends UnionSetMap[MultiPickListMap]
