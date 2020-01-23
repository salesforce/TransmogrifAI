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

import org.apache.spark.ml.linalg.Vector

import scala.reflect.runtime.universe._

/**
 * Factory for creating Feature Type instances
 *
 * @tparam T feature type
 */
sealed trait FeatureTypeFactory[T <: FeatureType] extends Serializable {

  /**
   * Make an instance of feature type with a given value.
   *
   * @param value value
   * @return feature type containing the value
   * @throws ClassCastException if the value type does not match the expected type
   * @throws IllegalArgumentException if unknown value type is specified
   */
  def newInstance(value: Any): T

}

/**
 * Factory for creating Feature Type instances from primitive values
 */
case object FeatureTypeFactory {
  /**
   * Factory for creating feature type instances
   *
   * @tparam T feature type
   * @return feature type factory
   */
  def apply[T <: FeatureType : WeakTypeTag](): FeatureTypeFactory[T] =
    new FeatureTypeFactory[T] {
      val maker: Any => T = newInstanceFn[T]
      def newInstance(value: Any): T = maker(value)
    }

  /**
   * Creates a function to make new feature type instances
   *
   * @tparam T feature type
   * @return a function to make new feature type instances
   */
  private def newInstanceFn[T <: FeatureType : WeakTypeTag]: Any => T = {
    val res = weakTypeOf[T] match {
      // Vector
      case t if t =:= weakTypeOf[OPVector] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.OPVector else new OPVector(value.asInstanceOf[Vector])

      // Lists
      case t if t =:= weakTypeOf[TextList] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.TextList else new TextList(value.asInstanceOf[Seq[String]])
      case t if t =:= weakTypeOf[DateList] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.DateList else new DateList(value.asInstanceOf[Seq[Long]])
      case t if t =:= weakTypeOf[DateTimeList] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.DateTimeList else new DateTimeList(value.asInstanceOf[Seq[Long]])
      case t if t =:= weakTypeOf[Geolocation] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Geolocation else new Geolocation(value.asInstanceOf[Seq[Double]])
      case t if t =:= weakTypeOf[MapList] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.MapList else new MapList(value.asInstanceOf[Seq[TextMap]])


      // Maps
      case t if t =:= weakTypeOf[Base64Map] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Base64Map else new Base64Map(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[BinaryMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.BinaryMap else new BinaryMap(value.asInstanceOf[Map[String, Boolean]])
      case t if t =:= weakTypeOf[ComboBoxMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.ComboBoxMap else new ComboBoxMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[CurrencyMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.CurrencyMap else new CurrencyMap(value.asInstanceOf[Map[String, Double]])
      case t if t =:= weakTypeOf[DateMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.DateMap else new DateMap(value.asInstanceOf[Map[String, Long]])
      case t if t =:= weakTypeOf[DateTimeMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.DateTimeMap else new DateTimeMap(value.asInstanceOf[Map[String, Long]])
      case t if t =:= weakTypeOf[EmailMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.EmailMap else new EmailMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[IDMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.IDMap else new IDMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[IntegralMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.IntegralMap else new IntegralMap(value.asInstanceOf[Map[String, Long]])
      case t if t =:= weakTypeOf[NameStats] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.NameStats else new NameStats(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[MultiPickListMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.MultiPickListMap
        else new MultiPickListMap(value.asInstanceOf[Map[String, Set[String]]])
      case t if t =:= weakTypeOf[PercentMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.PercentMap else new PercentMap(value.asInstanceOf[Map[String, Double]])
      case t if t =:= weakTypeOf[PhoneMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.PhoneMap else new PhoneMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[PickListMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.PickListMap else new PickListMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[RealMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.RealMap else new RealMap(value.asInstanceOf[Map[String, Double]])
      case t if t =:= weakTypeOf[TextAreaMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.TextAreaMap else new TextAreaMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[TextMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.TextMap else new TextMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[URLMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.URLMap else new URLMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[CountryMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.CountryMap else new CountryMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[StateMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.StateMap else new StateMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[CityMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.CityMap else new CityMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[PostalCodeMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.PostalCodeMap
        else new PostalCodeMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[StreetMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.StreetMap
        else new StreetMap(value.asInstanceOf[Map[String, String]])
      case t if t =:= weakTypeOf[GeolocationMap] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.GeolocationMap
        else new GeolocationMap(value.asInstanceOf[Map[String, Seq[Double]]])
      case t if t =:= weakTypeOf[Prediction] => (value: Any) =>
        new Prediction(value.asInstanceOf[Map[String, Double]])

      // Numerics
      case t if t =:= weakTypeOf[Binary] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Binary else new Binary(value.asInstanceOf[Option[Boolean]])
      case t if t =:= weakTypeOf[Currency] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Currency else new Currency(value.asInstanceOf[Option[Double]])
      case t if t =:= weakTypeOf[Date] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Date else new Date(value.asInstanceOf[Option[Long]])
      case t if t =:= weakTypeOf[DateTime] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.DateTime else new DateTime(value.asInstanceOf[Option[Long]])
      case t if t =:= weakTypeOf[Integral] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Integral else new Integral(value.asInstanceOf[Option[Long]])
      case t if t =:= weakTypeOf[Percent] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Percent else new Percent(value.asInstanceOf[Option[Double]])
      case t if t =:= weakTypeOf[Real] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Real else new Real(value.asInstanceOf[Option[Double]])
      case t if t =:= weakTypeOf[RealNN] => (value: Any) =>
        new RealNN(value.asInstanceOf[Option[Double]])

      // Sets
      case t if t =:= weakTypeOf[MultiPickList] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.MultiPickList else new MultiPickList(value.asInstanceOf[Set[String]])

      // Text
      case t if t =:= weakTypeOf[Base64] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Base64 else new Base64(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[ComboBox] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.ComboBox else new ComboBox(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[Email] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Email else new Email(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[ID] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.ID else new ID(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[Phone] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Phone else new Phone(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[PickList] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.PickList else new PickList(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[Text] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Text else new Text(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[TextArea] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.TextArea else new TextArea(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[URL] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.URL else new URL(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[Country] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Country else new Country(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[State] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.State else new State(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[City] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.City else new City(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[PostalCode] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.PostalCode else new PostalCode(value.asInstanceOf[Option[String]])
      case t if t =:= weakTypeOf[Street] => (value: Any) =>
        if (value == null) FeatureTypeDefaults.Street else new Street(value.asInstanceOf[Option[String]])

      // Unknown
      case t => throw new IllegalArgumentException(s"No feature type available for type $t")
    }
    res.asInstanceOf[Any => T]
  }
}
