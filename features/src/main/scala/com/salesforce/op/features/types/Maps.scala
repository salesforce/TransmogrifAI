/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types


class TextMap(val value: Map[String, String]) extends OPMap[String]
object TextMap {
  def apply(value: Map[String, String]): TextMap = new TextMap(value)
  def empty: TextMap = FeatureTypeDefaults.TextMap
}

class EmailMap(val value: Map[String, String]) extends OPMap[String]
object EmailMap {
  def apply(value: Map[String, String]): EmailMap = new EmailMap(value)
  def empty: EmailMap = FeatureTypeDefaults.EmailMap
}

class Base64Map(val value: Map[String, String]) extends OPMap[String]
object Base64Map {
  def apply(value: Map[String, String]): Base64Map = new Base64Map(value)
  def empty: Base64Map = FeatureTypeDefaults.Base64Map
}

class PhoneMap(val value: Map[String, String]) extends OPMap[String]
object PhoneMap {
  def apply(value: Map[String, String]): PhoneMap = new PhoneMap(value)
  def empty: PhoneMap = FeatureTypeDefaults.PhoneMap
}

class IDMap(val value: Map[String, String]) extends OPMap[String]
object IDMap {
  def apply(value: Map[String, String]): IDMap = new IDMap(value)
  def empty: IDMap = FeatureTypeDefaults.IDMap
}

class URLMap(val value: Map[String, String]) extends OPMap[String]
object URLMap {
  def apply(value: Map[String, String]): URLMap = new URLMap(value)
  def empty: URLMap = FeatureTypeDefaults.URLMap
}

class TextAreaMap(val value: Map[String, String]) extends OPMap[String]
object TextAreaMap {
  def apply(value: Map[String, String]): TextAreaMap = new TextAreaMap(value)
  def empty: TextAreaMap = FeatureTypeDefaults.TextAreaMap
}

class PickListMap(val value: Map[String, String]) extends OPMap[String]
object PickListMap {
  def apply(value: Map[String, String]): PickListMap = new PickListMap(value)
  def empty: PickListMap = FeatureTypeDefaults.PickListMap
}

class ComboBoxMap(val value: Map[String, String]) extends OPMap[String]
object ComboBoxMap {
  def apply(value: Map[String, String]): ComboBoxMap = new ComboBoxMap(value)
  def empty: ComboBoxMap = FeatureTypeDefaults.ComboBoxMap
}

class BinaryMap(val value: Map[String, Boolean]) extends OPMap[Boolean]
object BinaryMap {
  def apply(value: Map[String, Boolean]): BinaryMap = new BinaryMap(value)
  def empty: BinaryMap = FeatureTypeDefaults.BinaryMap
}

class IntegralMap(val value: Map[String, Long]) extends OPMap[Long]
object IntegralMap {
  def apply(value: Map[String, Long]): IntegralMap = new IntegralMap(value)
  def empty: IntegralMap = FeatureTypeDefaults.IntegralMap
}

class RealMap(val value: Map[String, Double]) extends OPMap[Double]
object RealMap {
  def apply(value: Map[String, Double]): RealMap = new RealMap(value)
  def empty: RealMap = FeatureTypeDefaults.RealMap
}

class PercentMap(val value: Map[String, Double]) extends OPMap[Double]
object PercentMap {
  def apply(value: Map[String, Double]): PercentMap = new PercentMap(value)
  def empty: PercentMap = FeatureTypeDefaults.PercentMap
}

class CurrencyMap(val value: Map[String, Double]) extends OPMap[Double]
object CurrencyMap {
  def apply(value: Map[String, Double]): CurrencyMap = new CurrencyMap(value)
  def empty: CurrencyMap = FeatureTypeDefaults.CurrencyMap
}

class DateMap(val value: Map[String, Long]) extends OPMap[Long]
object DateMap {
  def apply(value: Map[String, Long]): DateMap = new DateMap(value)
  def empty: DateMap = FeatureTypeDefaults.DateMap
}
class DateTimeMap(val value: Map[String, Long]) extends OPMap[Long]
object DateTimeMap {
  def apply(value: Map[String, Long]): DateTimeMap = new DateTimeMap(value)
  def empty: DateTimeMap = FeatureTypeDefaults.DateTimeMap
}

class MultiPickListMap(val value: Map[String, Set[String]]) extends OPMap[Set[String]]
object MultiPickListMap {
  def apply(value: Map[String, Set[String]]): MultiPickListMap = new MultiPickListMap(value)
  def empty: MultiPickListMap = FeatureTypeDefaults.MultiPickListMap
}

class CountryMap(val value: Map[String, String]) extends OPMap[String] with Location
object CountryMap {
  def apply(value: Map[String, String]): CountryMap = new CountryMap(value)
  def empty: CountryMap = FeatureTypeDefaults.CountryMap
}

class StateMap(val value: Map[String, String]) extends OPMap[String] with Location
object StateMap {
  def apply(value: Map[String, String]): StateMap = new StateMap(value)
  def empty: StateMap = FeatureTypeDefaults.StateMap
}

class CityMap(val value: Map[String, String]) extends OPMap[String] with Location
object CityMap {
  def apply(value: Map[String, String]): CityMap = new CityMap(value)
  def empty: CityMap = FeatureTypeDefaults.CityMap
}

class PostalCodeMap(val value: Map[String, String]) extends OPMap[String] with Location
object PostalCodeMap {
  def apply(value: Map[String, String]): PostalCodeMap = new PostalCodeMap(value)
  def empty: PostalCodeMap = FeatureTypeDefaults.PostalCodeMap
}

class StreetMap(val value: Map[String, String]) extends OPMap[String] with Location
object StreetMap {
  def apply(value: Map[String, String]): StreetMap = new StreetMap(value)
  def empty: StreetMap = FeatureTypeDefaults.StreetMap
}

class GeolocationMap(val value: Map[String, Seq[Double]]) extends OPMap[Seq[Double]] with Location
object GeolocationMap {
  def apply(value: Map[String, Seq[Double]]): GeolocationMap = new GeolocationMap(value)
  def empty: GeolocationMap = FeatureTypeDefaults.GeolocationMap
}
