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
import org.apache.spark.ml.linalg.Vectors

import scala.reflect.runtime.universe._

/**
 * Default values for Feature Types
 */
case object FeatureTypeDefaults {

  // Numerics
  val Binary = new t.Binary(None)
  val Integral = new t.Integral(None)
  val Real = new t.Real(None)
  val Date = new t.Date(None)
  val DateTime = new t.DateTime(None)
  val Currency = new t.Currency(None)
  val Percent = new t.Percent(None)

  // Text
  val Text = new t.Text(None)
  val Base64 = new t.Base64(None)
  val ComboBox = new t.ComboBox(None)
  val Email = new t.Email(None)
  val ID = new t.ID(None)
  val Phone = new t.Phone(None)
  val PickList = new t.PickList(None)
  val TextArea = new t.TextArea(None)
  val URL = new t.URL(None)
  val Country = new t.Country(None)
  val State = new t.State(None)
  val Street = new t.Street(None)
  val City = new t.City(None)
  val PostalCode = new t.PostalCode(None)

  // Vector
  val OPVector = new t.OPVector(Vectors.zeros(0))

  // Lists
  val TextList = new t.TextList(Seq.empty)
  val DateList = new t.DateList(Seq.empty)
  val DateTimeList = new t.DateTimeList(Seq.empty)
  val Geolocation = new t.Geolocation(Seq.empty)

  // Sets
  val MultiPickList = new t.MultiPickList(Set.empty)

  // Maps
  val Base64Map = new t.Base64Map(Map.empty)
  val BinaryMap = new t.BinaryMap(Map.empty)
  val ComboBoxMap = new t.ComboBoxMap(Map.empty)
  val CurrencyMap = new t.CurrencyMap(Map.empty)
  val DateMap = new t.DateMap(Map.empty)
  val DateTimeMap = new t.DateTimeMap(Map.empty)
  val EmailMap = new t.EmailMap(Map.empty)
  val IDMap = new t.IDMap(Map.empty)
  val IntegralMap = new t.IntegralMap(Map.empty)
  val MultiPickListMap = new t.MultiPickListMap(Map.empty)
  val PercentMap = new t.PercentMap(Map.empty)
  val PhoneMap = new t.PhoneMap(Map.empty)
  val PickListMap = new t.PickListMap(Map.empty)
  val RealMap = new t.RealMap(Map.empty)
  val TextAreaMap = new t.TextAreaMap(Map.empty)
  val TextMap = new t.TextMap(Map.empty)
  val URLMap = new t.URLMap(Map.empty)
  val CountryMap = new t.CountryMap(Map.empty)
  val StateMap = new t.StateMap(Map.empty)
  val CityMap = new t.CityMap(Map.empty)
  val PostalCodeMap = new t.PostalCodeMap(Map.empty)
  val StreetMap = new t.StreetMap(Map.empty)
  val NameStats = new t.NameStats(Map.empty)
  val GeolocationMap = new t.GeolocationMap(Map.empty)

  /**
   * Return a default value for specified feature type
   *
   * Note: reflection lookups are VERY slow. use with caution!
   *
   * @tparam O feature type
   * @return a default value for specified feature type
   */
  private[op] def default[O <: FeatureType : WeakTypeTag]: O = {
    val value = weakTypeOf[O] match {
      // Vector
      case wt if wt =:= weakTypeOf[t.OPVector] => OPVector

      // Lists
      case wt if wt =:= weakTypeOf[t.TextList] => TextList
      case wt if wt =:= weakTypeOf[t.DateList] => DateList
      case wt if wt =:= weakTypeOf[t.DateTimeList] => DateTimeList
      case wt if wt =:= weakTypeOf[t.Geolocation] => Geolocation

      // Maps
      case wt if wt =:= weakTypeOf[t.Base64Map] => Base64Map
      case wt if wt =:= weakTypeOf[t.BinaryMap] => BinaryMap
      case wt if wt =:= weakTypeOf[t.ComboBoxMap] => ComboBoxMap
      case wt if wt =:= weakTypeOf[t.CurrencyMap] => CurrencyMap
      case wt if wt =:= weakTypeOf[t.DateMap] => DateMap
      case wt if wt =:= weakTypeOf[t.DateTimeMap] => DateTimeMap
      case wt if wt =:= weakTypeOf[t.EmailMap] => EmailMap
      case wt if wt =:= weakTypeOf[t.IDMap] => IDMap
      case wt if wt =:= weakTypeOf[t.IntegralMap] => IntegralMap
      case wt if wt =:= weakTypeOf[t.MultiPickListMap] => MultiPickListMap
      case wt if wt =:= weakTypeOf[t.PercentMap] => PercentMap
      case wt if wt =:= weakTypeOf[t.PhoneMap] => PhoneMap
      case wt if wt =:= weakTypeOf[t.PickListMap] => PickListMap
      case wt if wt =:= weakTypeOf[t.RealMap] => RealMap
      case wt if wt =:= weakTypeOf[t.TextAreaMap] => TextAreaMap
      case wt if wt =:= weakTypeOf[t.TextMap] => TextMap
      case wt if wt =:= weakTypeOf[t.URLMap] => URLMap
      case wt if wt =:= weakTypeOf[t.CountryMap] => CountryMap
      case wt if wt =:= weakTypeOf[t.StateMap] => StateMap
      case wt if wt =:= weakTypeOf[t.CityMap] => CityMap
      case wt if wt =:= weakTypeOf[t.PostalCodeMap] => PostalCodeMap
      case wt if wt =:= weakTypeOf[t.StreetMap] => StreetMap
      case wt if wt =:= weakTypeOf[t.NameStats] => NameStats
      case wt if wt =:= weakTypeOf[t.GeolocationMap] => GeolocationMap

      // Numerics
      case wt if wt =:= weakTypeOf[t.Binary] => Binary
      case wt if wt =:= weakTypeOf[t.Currency] => Currency
      case wt if wt =:= weakTypeOf[t.Date] => Date
      case wt if wt =:= weakTypeOf[t.DateTime] => DateTime
      case wt if wt =:= weakTypeOf[t.Integral] => Integral
      case wt if wt =:= weakTypeOf[t.Percent] => Percent
      case wt if wt =:= weakTypeOf[t.Real] => Real

      // Sets
      case wt if wt =:= weakTypeOf[t.MultiPickList] => MultiPickList

      // Text
      case wt if wt =:= weakTypeOf[t.Base64] => Base64
      case wt if wt =:= weakTypeOf[t.ComboBox] => ComboBox
      case wt if wt =:= weakTypeOf[t.Email] => Email
      case wt if wt =:= weakTypeOf[t.ID] => ID
      case wt if wt =:= weakTypeOf[t.Phone] => Phone
      case wt if wt =:= weakTypeOf[t.PickList] => PickList
      case wt if wt =:= weakTypeOf[t.Text] => Text
      case wt if wt =:= weakTypeOf[t.TextArea] => TextArea
      case wt if wt =:= weakTypeOf[t.URL] => URL
      case wt if wt =:= weakTypeOf[t.Country] => Country
      case wt if wt =:= weakTypeOf[t.State] => State
      case wt if wt =:= weakTypeOf[t.City] => City
      case wt if wt =:= weakTypeOf[t.PostalCode] => PostalCode
      case wt if wt =:= weakTypeOf[t.Street] => Street

      // Unknown
      case wt => throw new IllegalArgumentException(s"No default value available for feature type $wt")
    }
    value.asInstanceOf[O]
  }

}
