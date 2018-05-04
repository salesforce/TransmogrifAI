/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import com.salesforce.op.features.{types => t}

import scala.collection.mutable.{WrappedArray => MWrappedArray}
import scala.reflect.runtime.universe._


/**
 * Feature type from/to Spark primitives converter, i.e Real from/to Double etc.
 *
 * @tparam T feature type
 */
sealed trait FeatureTypeSparkConverter[T <: FeatureType] extends Serializable {

  /**
   * Feature type factory instance
   */
  val ftFactory: FeatureTypeFactory[T]

  /**
   * Converts a primitive Spark value into a feature type
   *
   * @param value a primitive Spark value
   * @return feature type instance
   */
  def fromSpark(value: Any): T

  /**
   * Converts feature type into a Spark primitive value
   *
   * @param ft feature type instance
   * @return primitive value
   */
  def toSpark(ft: T): Any = FeatureTypeSparkConverter.toSpark(ft)
}

/**
 * Feature type from/to Spark primitives converter, i.e Real from/to Double etc.
 */
case object FeatureTypeSparkConverter {
  /**
   * Feature type from/to Spark primitives converter, i.e Real from/to Double etc.
   *
   * @tparam T feature type
   * @return feature type from/to Spark primitives converter
   */
  def apply[T <: FeatureType : WeakTypeTag](): FeatureTypeSparkConverter[T] =
    new FeatureTypeSparkConverter[T] {
      val ftFactory = FeatureTypeFactory[T]()
      val maker: Any => T = fromSparkFn[T](ftFactory)
      def fromSpark(value: Any): T = maker(value)
    }


  /**
   * Converts feature type into a Spark primitive value
   *
   * @param ft feature type instance
   * @return primitive value
   */
  def toSpark(ft: FeatureType): Any = ft match {
    // Empty check
    case n if n.isEmpty => null
    // Text
    case n: t.Text => n.value.get
    // Numerics
    case n: t.OPNumeric[_] => n.value.get
    // Vector
    case n: t.OPVector => n.value
    // Lists
    case n: t.OPList[_] => MWrappedArray.make(n.toArray)
    // Sets
    case n: t.OPSet[_] => MWrappedArray.make(n.toArray)
    // Maps
    case n: t.MultiPickListMap => n.value.map { case (k, v) => k -> MWrappedArray.make(v.toArray) }
    case n: t.GeolocationMap => n.value.map { case (k, v) => k -> MWrappedArray.make(v.toArray) }
    case n: t.OPMap[_] => n.value
  }

  /**
   * Creates a function to convert a primitive Spark value into a feature type
   *
   * @param ftFactory feature type factory
   * @tparam O feature type
   * @return a function to convert a primitive Spark value into a feature type
   */
  private def fromSparkFn[O <: FeatureType : WeakTypeTag](ftFactory: FeatureTypeFactory[O]): Any => O = {
    val wrapped: Any => Any = weakTypeOf[O] match {
      // Text
      case wt if wt <:< weakTypeOf[t.Text] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Text.value else Some(value.asInstanceOf[String])

      // Numerics
      case wt if wt <:< weakTypeOf[t.RealNN] => (value: Any) =>
        if (value == null) None else Some(value.asInstanceOf[Double])
      case wt if wt <:< weakTypeOf[t.Real] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Real.value else Some(value.asInstanceOf[Double])
      case wt if wt <:< weakTypeOf[t.Integral] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Integral.value else Some(value.asInstanceOf[Long])
      case wt if wt <:< weakTypeOf[t.Binary] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Binary.value else Some(value.asInstanceOf[Boolean])

      // Maps
      case wt if wt <:< weakTypeOf[t.MultiPickListMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.MultiPickListMap.value
        else value.asInstanceOf[Map[String, MWrappedArray[String]]].map { case (k, v) => k -> v.toSet }

      // Sets
      case wt if wt <:< weakTypeOf[t.MultiPickList] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.MultiPickList.value
        else value.asInstanceOf[MWrappedArray[String]].toSet

      // Everything else
      case _ => identity
    }
    (value: Any) => ftFactory.newInstance(wrapped(value))
  }
}

/**
 * All feature type from/to Spark primitives converters
 */
trait FeatureTypeSparkConverters {
  // Numerics
  implicit val BinaryConverter = FeatureTypeSparkConverter[Binary]()
  implicit val IntegralConverter = FeatureTypeSparkConverter[Integral]()
  implicit val RealConverter = FeatureTypeSparkConverter[Real]()
  implicit val RealNNConverter = FeatureTypeSparkConverter[RealNN]()
  implicit val DateConverter = FeatureTypeSparkConverter[Date]()
  implicit val DateTimeConverter = FeatureTypeSparkConverter[DateTime]()
  implicit val CurrencyConverter = FeatureTypeSparkConverter[Currency]()
  implicit val PercentConverter = FeatureTypeSparkConverter[Percent]()

  // Text
  implicit val TextConverter = FeatureTypeSparkConverter[Text]()
  implicit val Base64Converter = FeatureTypeSparkConverter[Base64]()
  implicit val ComboBoxConverter = FeatureTypeSparkConverter[ComboBox]()
  implicit val EmailConverter = FeatureTypeSparkConverter[Email]()
  implicit val IDConverter = FeatureTypeSparkConverter[ID]()
  implicit val PhoneConverter = FeatureTypeSparkConverter[Phone]()
  implicit val PickListConverter = FeatureTypeSparkConverter[PickList]()
  implicit val TextAreaConverter = FeatureTypeSparkConverter[TextArea]()
  implicit val URLConverter = FeatureTypeSparkConverter[URL]()
  implicit val CountryConverter = FeatureTypeSparkConverter[Country]()
  implicit val StateConverter = FeatureTypeSparkConverter[State]()
  implicit val CityConverter = FeatureTypeSparkConverter[City]()
  implicit val PostalCodeConverter = FeatureTypeSparkConverter[PostalCode]()
  implicit val StreetConverter = FeatureTypeSparkConverter[Street]()

  // Vector
  implicit val OPVectorConverter = FeatureTypeSparkConverter[OPVector]()

  // Lists
  implicit val TextListConverter = FeatureTypeSparkConverter[TextList]()
  implicit val DateListConverter = FeatureTypeSparkConverter[DateList]()
  implicit val DateTimeListConverter = FeatureTypeSparkConverter[DateTimeList]()
  implicit val GeolocationConverter = FeatureTypeSparkConverter[Geolocation]()

  // Sets
  implicit val MultiPickListConverter = FeatureTypeSparkConverter[MultiPickList]()

  // Maps
  implicit val Base64MapConverter = FeatureTypeSparkConverter[Base64Map]()
  implicit val BinaryMapConverter = FeatureTypeSparkConverter[BinaryMap]()
  implicit val ComboBoxMapConverter = FeatureTypeSparkConverter[ComboBoxMap]()
  implicit val CurrencyMapConverter = FeatureTypeSparkConverter[CurrencyMap]()
  implicit val DateMapConverter = FeatureTypeSparkConverter[DateMap]()
  implicit val DateTimeMapConverter = FeatureTypeSparkConverter[DateTimeMap]()
  implicit val EmailMapConverter = FeatureTypeSparkConverter[EmailMap]()
  implicit val IDMapConverter = FeatureTypeSparkConverter[IDMap]()
  implicit val IntegralMapConverter = FeatureTypeSparkConverter[IntegralMap]()
  implicit val MultiPickListMapConverter = FeatureTypeSparkConverter[MultiPickListMap]()
  implicit val PercentMapConverter = FeatureTypeSparkConverter[PercentMap]()
  implicit val PhoneMapConverter = FeatureTypeSparkConverter[PhoneMap]()
  implicit val PickListMapConverter = FeatureTypeSparkConverter[PickListMap]()
  implicit val RealMapConverter = FeatureTypeSparkConverter[RealMap]()
  implicit val TextAreaMapConverter = FeatureTypeSparkConverter[TextAreaMap]()
  implicit val TextMapConverter = FeatureTypeSparkConverter[TextMap]()
  implicit val URLMapConverter = FeatureTypeSparkConverter[URLMap]()
  implicit val CountryMapConverter = FeatureTypeSparkConverter[CountryMap]()
  implicit val StateMapConverter = FeatureTypeSparkConverter[StateMap]()
  implicit val CityMapConverter = FeatureTypeSparkConverter[CityMap]()
  implicit val PostalCodeMapConverter = FeatureTypeSparkConverter[PostalCodeMap]()
  implicit val StreetMapConverter = FeatureTypeSparkConverter[StreetMap]()
  implicit val GeolocationMapConverter = FeatureTypeSparkConverter[GeolocationMap]()
  implicit val PredictionConverter = FeatureTypeSparkConverter[Prediction]()
}

/**
 * All feature type from/to Spark primitives converters
 */
case object FeatureTypeSparkConverters extends FeatureTypeSparkConverters
