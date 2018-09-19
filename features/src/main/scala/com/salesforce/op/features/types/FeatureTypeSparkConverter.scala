/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.features.types

import com.salesforce.op.features.{types => t}

import scala.collection.mutable.{WrappedArray => MWrappedArray}
import scala.reflect.runtime.universe._


/**
 * Feature Type from/to Spark primitives converter, i.e Real from/to Double etc.
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
   * For a given feature type class (or [[FeatureType.typeName]]) from/to Spark primitives converter,
   * i.e Real from/to Double etc.
   *
   * @param featureTypeName full class name of the feature type, see [[FeatureType.typeName]]
   * @throws IllegalArgumentException if feature type name is unknown
   * @return feature type from/to Spark primitives converter
   */
  def fromFeatureTypeName(featureTypeName: String): FeatureTypeSparkConverter[_ <: FeatureType] = {
    featureTypeSparkConverters.get(featureTypeName) match {
      case Some(converter) => converter
      case None => throw new IllegalArgumentException(s"Unknown feature type '$featureTypeName'")
    }
  }

  /**
   * A map from feature type class to [[FeatureTypeSparkConverter]]
   */
  private[types] val featureTypeSparkConverters: Map[String, FeatureTypeSparkConverter[_ <: FeatureType]] =
    FeatureType.featureTypeTags.map {
      case (featureTypeClass, featureTypeTag) =>
        featureTypeClass.getName ->
          FeatureTypeSparkConverter[FeatureType]()(featureTypeTag.asInstanceOf[WeakTypeTag[FeatureType]])
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
        value match {
          case null => None
          case _: Float => Some(value.asInstanceOf[Float].toDouble)
          case _: Double => Some(value.asInstanceOf[Double])
          case _ => throw new IllegalArgumentException(s"RealNN type mapping is not defined for ${value.getClass}")
        }
      case wt if wt <:< weakTypeOf[t.Real] => (value: Any) =>
        value match {
          case null => FeatureTypeDefaults.Real.value
          case _: Float => Some(value.asInstanceOf[Float].toDouble)
          case _: Double => Some(value.asInstanceOf[Double])
          case _ => throw new IllegalArgumentException(s"Real type mapping is not defined for ${value.getClass}")
        }
      case wt if wt <:< weakTypeOf[t.Integral] => (value: Any) =>
        value match {
          case null => FeatureTypeDefaults.Integral.value
          case _: Short => Some(value.asInstanceOf[Short].toLong)
          case _: Int => Some(value.asInstanceOf[Int].toLong)
          case _: Long => Some(value.asInstanceOf[Long])
          case _ => throw new IllegalArgumentException(s"Integral type mapping is not defined for ${value.getClass}")
        }
      case wt if wt <:< weakTypeOf[t.Binary] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Binary.value else Some(value.asInstanceOf[Boolean])

      // Date & Time
      case wt if wt <:< weakTypeOf[t.Date] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Date.value else Some(value.asInstanceOf[Int].toLong)
      case wt if wt <:< weakTypeOf[t.DateTime] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.DateTime.value else Some(value.asInstanceOf[Long])

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
