/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features

import com.salesforce.op.utils.spark.RichRow._
import com.salesforce.op.aggregators._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.{FeatureGeneratorStage, OpPipelineStage}
import com.twitter.algebird.MonoidAggregator
import org.apache.spark.sql.Row
import org.joda.time.Duration

import scala.language.experimental.macros
import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Factory for creating features
 */
object FeatureBuilder {
  // scalastyle:off

  // Vector
  def OPVector[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, OPVector] = OPVector(name.value)

  // Lists
  def TextList[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, TextList] = TextList(name.value)
  def DateList[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, DateList] = DateList(name.value)
  def DateTimeList[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, DateTimeList] = DateTimeList(name.value)
  def Geolocation[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Geolocation] = Geolocation(name.value)

  // Maps
  def Base64Map[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Base64Map] = Base64Map(name.value)
  def BinaryMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, BinaryMap] = BinaryMap(name.value)
  def ComboBoxMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, ComboBoxMap] = ComboBoxMap(name.value)
  def CurrencyMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, CurrencyMap] = CurrencyMap(name.value)
  def DateMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, DateMap] = DateMap(name.value)
  def DateTimeMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, DateTimeMap] = DateTimeMap(name.value)
  def EmailMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, EmailMap] = EmailMap(name.value)
  def IDMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, IDMap] = IDMap(name.value)
  def IntegralMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, IntegralMap] = IntegralMap(name.value)
  def MultiPickListMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, MultiPickListMap] = MultiPickListMap(name.value)
  def PercentMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, PercentMap] = PercentMap(name.value)
  def PhoneMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, PhoneMap] = PhoneMap(name.value)
  def PickListMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, PickListMap] = PickListMap(name.value)
  def RealMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, RealMap] = RealMap(name.value)
  def TextAreaMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, TextAreaMap] = TextAreaMap(name.value)
  def TextMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, TextMap] = TextMap(name.value)
  def URLMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, URLMap] = URLMap(name.value)
  def CountryMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, CountryMap] = CountryMap(name.value)
  def StateMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, StateMap] = StateMap(name.value)
  def CityMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, CityMap] = CityMap(name.value)
  def PostalCodeMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, PostalCodeMap] = PostalCodeMap(name.value)
  def StreetMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, StreetMap] = StreetMap(name.value)
  def GeolocationMap[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, GeolocationMap] = GeolocationMap(name.value)
  def Prediction[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Prediction] = Prediction(name.value)

  // Numerics
  def Binary[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Binary] = Binary(name.value)
  def Currency[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Currency] = Currency(name.value)
  def Date[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Date] = Date(name.value)
  def DateTime[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, DateTime] = DateTime(name.value)
  def Integral[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Integral] = Integral(name.value)
  def Percent[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Percent] = Percent(name.value)
  def Real[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Real] = Real(name.value)
  def RealNN[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, RealNN] = RealNN(name.value)

  // Sets
  def MultiPickList[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, MultiPickList] = MultiPickList(name.value)

  // Text
  def Base64[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Base64] = Base64(name.value)
  def ComboBox[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, ComboBox] = ComboBox(name.value)
  def Email[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Email] = Email(name.value)
  def ID[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, ID] = ID(name.value)
  def Phone[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Phone] = Phone(name.value)
  def PickList[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, PickList] = PickList(name.value)
  def Text[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Text] = Text(name.value)
  def TextArea[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, TextArea] = TextArea(name.value)
  def URL[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, URL] = URL(name.value)
  def Country[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Country] = Country(name.value)
  def State[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, State] = State(name.value)
  def City[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, City] = City(name.value)
  def PostalCode[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, PostalCode] = PostalCode(name.value)
  def Street[I: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilder[I, Street] = Street(name.value)

  // Vector
  def OPVector[I: WeakTypeTag](name: String): FeatureBuilder[I, OPVector] = FeatureBuilder[I, OPVector](name = name)

  // Lists
  def TextList[I: WeakTypeTag](name: String): FeatureBuilder[I, TextList] = FeatureBuilder[I, TextList](name = name)
  def DateList[I: WeakTypeTag](name: String): FeatureBuilder[I, DateList] = FeatureBuilder[I, DateList](name = name)
  def DateTimeList[I: WeakTypeTag](name: String): FeatureBuilder[I, DateTimeList] = FeatureBuilder[I, DateTimeList](name = name)
  def Geolocation[I: WeakTypeTag](name: String): FeatureBuilder[I, Geolocation] = FeatureBuilder[I, Geolocation](name = name)

  // Maps
  def Base64Map[I: WeakTypeTag](name: String): FeatureBuilder[I, Base64Map] = FeatureBuilder[I, Base64Map](name = name)
  def BinaryMap[I: WeakTypeTag](name: String): FeatureBuilder[I, BinaryMap] = FeatureBuilder[I, BinaryMap](name = name)
  def ComboBoxMap[I: WeakTypeTag](name: String): FeatureBuilder[I, ComboBoxMap] = FeatureBuilder[I, ComboBoxMap](name = name)
  def CurrencyMap[I: WeakTypeTag](name: String): FeatureBuilder[I, CurrencyMap] = FeatureBuilder[I, CurrencyMap](name = name)
  def DateMap[I: WeakTypeTag](name: String): FeatureBuilder[I, DateMap] = FeatureBuilder[I, DateMap](name = name)
  def DateTimeMap[I: WeakTypeTag](name: String): FeatureBuilder[I, DateTimeMap] = FeatureBuilder[I, DateTimeMap](name = name)
  def EmailMap[I: WeakTypeTag](name: String): FeatureBuilder[I, EmailMap] = FeatureBuilder[I, EmailMap](name = name)
  def IDMap[I: WeakTypeTag](name: String): FeatureBuilder[I, IDMap] = FeatureBuilder[I, IDMap](name = name)
  def IntegralMap[I: WeakTypeTag](name: String): FeatureBuilder[I, IntegralMap] = FeatureBuilder[I, IntegralMap](name = name)
  def MultiPickListMap[I: WeakTypeTag](name: String): FeatureBuilder[I, MultiPickListMap] = FeatureBuilder[I, MultiPickListMap](name = name)
  def PercentMap[I: WeakTypeTag](name: String): FeatureBuilder[I, PercentMap] = FeatureBuilder[I, PercentMap](name = name)
  def PhoneMap[I: WeakTypeTag](name: String): FeatureBuilder[I, PhoneMap] = FeatureBuilder[I, PhoneMap](name = name)
  def PickListMap[I: WeakTypeTag](name: String): FeatureBuilder[I, PickListMap] = FeatureBuilder[I, PickListMap](name = name)
  def RealMap[I: WeakTypeTag](name: String): FeatureBuilder[I, RealMap] = FeatureBuilder[I, RealMap](name = name)
  def TextAreaMap[I: WeakTypeTag](name: String): FeatureBuilder[I, TextAreaMap] = FeatureBuilder[I, TextAreaMap](name = name)
  def TextMap[I: WeakTypeTag](name: String): FeatureBuilder[I, TextMap] = FeatureBuilder[I, TextMap](name = name)
  def URLMap[I: WeakTypeTag](name: String): FeatureBuilder[I, URLMap] = FeatureBuilder[I, URLMap](name = name)
  def CountryMap[I: WeakTypeTag](name: String): FeatureBuilder[I, CountryMap] = FeatureBuilder[I, CountryMap](name = name)
  def StateMap[I: WeakTypeTag](name: String): FeatureBuilder[I, StateMap] = FeatureBuilder[I, StateMap](name = name)
  def CityMap[I: WeakTypeTag](name: String): FeatureBuilder[I, CityMap] = FeatureBuilder[I, CityMap](name = name)
  def PostalCodeMap[I: WeakTypeTag](name: String): FeatureBuilder[I, PostalCodeMap] = FeatureBuilder[I, PostalCodeMap](name = name)
  def StreetMap[I: WeakTypeTag](name: String): FeatureBuilder[I, StreetMap] = FeatureBuilder[I, StreetMap](name = name)
  def GeolocationMap[I: WeakTypeTag](name: String): FeatureBuilder[I, GeolocationMap] = FeatureBuilder[I, GeolocationMap](name = name)
  def Prediction[I: WeakTypeTag](name: String): FeatureBuilder[I, Prediction] = FeatureBuilder[I, Prediction](name = name)

  // Numerics
  def Binary[I: WeakTypeTag](name: String): FeatureBuilder[I, Binary] = FeatureBuilder[I, Binary](name = name)
  def Currency[I: WeakTypeTag](name: String): FeatureBuilder[I, Currency] = FeatureBuilder[I, Currency](name = name)
  def Date[I: WeakTypeTag](name: String): FeatureBuilder[I, Date] = FeatureBuilder[I, Date](name = name)
  def DateTime[I: WeakTypeTag](name: String): FeatureBuilder[I, DateTime] = FeatureBuilder[I, DateTime](name = name)
  def Integral[I: WeakTypeTag](name: String): FeatureBuilder[I, Integral] = FeatureBuilder[I, Integral](name = name)
  def Percent[I: WeakTypeTag](name: String): FeatureBuilder[I, Percent] = FeatureBuilder[I, Percent](name = name)
  def Real[I: WeakTypeTag](name: String): FeatureBuilder[I, Real] = FeatureBuilder[I, Real](name = name)
  def RealNN[I: WeakTypeTag](name: String): FeatureBuilder[I, RealNN] = FeatureBuilder[I, RealNN](name = name)

  // Sets
  def MultiPickList[I: WeakTypeTag](name: String): FeatureBuilder[I, MultiPickList] = FeatureBuilder[I, MultiPickList](name = name)

  // Text
  def Base64[I: WeakTypeTag](name: String): FeatureBuilder[I, Base64] = FeatureBuilder[I, Base64](name = name)
  def ComboBox[I: WeakTypeTag](name: String): FeatureBuilder[I, ComboBox] = FeatureBuilder[I, ComboBox](name = name)
  def Email[I: WeakTypeTag](name: String): FeatureBuilder[I, Email] = FeatureBuilder[I, Email](name = name)
  def ID[I: WeakTypeTag](name: String): FeatureBuilder[I, ID] = FeatureBuilder[I, ID](name = name)
  def Phone[I: WeakTypeTag](name: String): FeatureBuilder[I, Phone] = FeatureBuilder[I, Phone](name = name)
  def PickList[I: WeakTypeTag](name: String): FeatureBuilder[I, PickList] = FeatureBuilder[I, PickList](name = name)
  def Text[I: WeakTypeTag](name: String): FeatureBuilder[I, Text] = FeatureBuilder[I, Text](name = name)
  def TextArea[I: WeakTypeTag](name: String): FeatureBuilder[I, TextArea] = FeatureBuilder[I, TextArea](name = name)
  def URL[I: WeakTypeTag](name: String): FeatureBuilder[I, URL] = FeatureBuilder[I, URL](name = name)
  def Country[I: WeakTypeTag](name: String): FeatureBuilder[I, Country] = FeatureBuilder[I, Country](name = name)
  def State[I: WeakTypeTag](name: String): FeatureBuilder[I, State] = FeatureBuilder[I, State](name = name)
  def City[I: WeakTypeTag](name: String): FeatureBuilder[I, City] = FeatureBuilder[I, City](name = name)
  def PostalCode[I: WeakTypeTag](name: String): FeatureBuilder[I, PostalCode] = FeatureBuilder[I, PostalCode](name = name)
  def Street[I: WeakTypeTag](name: String): FeatureBuilder[I, Street] = FeatureBuilder[I, Street](name = name)

  def apply[I: WeakTypeTag, O <: FeatureType : WeakTypeTag](name: String): FeatureBuilder[I, O] = new FeatureBuilder[I, O](name)

  def fromRow[O <: FeatureType: WeakTypeTag](implicit name: sourcecode.Name): FeatureBuilderWithExtract[Row, O] = fromRow[O](name.value, None)
  def fromRow[O <: FeatureType: WeakTypeTag](name: String): FeatureBuilderWithExtract[Row, O] = fromRow[O](name, None)
  def fromRow[O <: FeatureType: WeakTypeTag](index: Int)(implicit name: sourcecode.Name): FeatureBuilderWithExtract[Row, O] = fromRow[O](name.value, Some(index))
  def fromRow[O <: FeatureType: WeakTypeTag](name: String, index: Option[Int]): FeatureBuilderWithExtract[Row, O] = {
    val c = FeatureTypeSparkConverter[O]()
    new FeatureBuilderWithExtract[Row, O](
      name = name,
      extractFn = (r: Row) => c.fromSpark(index.map(r.get).getOrElse(r.getAny(name))),
      extractSource = "(r: Row) => c.fromSpark(index.map(r.get).getOrElse(r.getAny(name)))"
    )
  }

  // scalastyle:on

}

/**
 * Feature Builder allows building features
 *
 * @param name feature name
 * @tparam I input data type
 * @tparam O output feature type
 */
final class FeatureBuilder[I, O <: FeatureType](val name: String) {

  /**
   * Feature extract method - a function to extract value of the feature from the raw data.
   *
   * @param fn a function to extract value of the feature from the raw data
   */
  def extract(fn: I => O): FeatureBuilderWithExtract[I, O] =
    macro FeatureBuilderMacros.extract[I, O]

  /**
   * Feature extract method - a function to extract value of the feature from the raw data.
   * If the function 'fn' throws an Exception the `default` value is returned instead.
   *
   * @param fn      a function to extract value of the feature from the raw data
   * @param default the default value
   */
  def extract(fn: I => O, default: O): FeatureBuilderWithExtract[I, O] =
    macro FeatureBuilderMacros.extractWithDefault[I, O]

}

/**
 * Feature Builder with an extract function
 *
 * @param name          feature name
 * @param extractFn     a function to extract value of the feature from the raw data
 * @param extractSource source code of the extract function
 */
final class FeatureBuilderWithExtract[I, O <: FeatureType]
(
  val name: String,
  val extractFn: I => O,
  val extractSource: String
)(implicit val tti: WeakTypeTag[I], val tto: WeakTypeTag[O]) {

  var aggregator: MonoidAggregator[Event[O], _, O] = MonoidAggregatorDefaults.aggregatorOf[O](tto)
  var aggregateWindow: Option[Duration] = None

  /**
   * Feature aggregation function with zero value
   * @param zero a zero element for aggregation
   * @param fn   aggregation function
   */
  def aggregate(zero: O#Value, fn: (O#Value, O#Value) => O#Value): this.type = {
    aggregator = new CustomMonoidAggregator[O](associativeFn = fn, zero = zero)(tto)
    this
  }

  /**
   * Feature aggregation function with zero value of [[FeatureTypeDefaults.default[O].value]]
   * @param fn aggregation function
   */
  def aggregate(fn: (O#Value, O#Value) => O#Value): this.type = {
    val zero = FeatureTypeDefaults.default[O](tto).value
    aggregator = new CustomMonoidAggregator[O](associativeFn = fn, zero = zero)(tto)
    this
  }

  /**
   * Feature aggregation with a monoid aggregator
   * @param monoid a monoid aggregator
   */
  def aggregate(monoid: MonoidAggregator[Event[O], _, O]): this.type = {
    aggregator = monoid
    this
  }

  /**
   * Aggregation time window
   * @param time a time period during which to include features in aggregation
   */
  def window(time: Duration): this.type = {
    aggregateWindow = Option(time)
    this
  }

  /**
   * Make a predictor feature
   * @return a predictor feature
   */
  def asPredictor: Feature[O] = makeFeature(isResponse = false)

  /**
   * Make a response feature
   * @return a response feature
   */
  def asResponse: Feature[O] = makeFeature(isResponse = true)

  private def makeFeature(isResponse: Boolean): Feature[O] = {
    val originStage: OpPipelineStage[O] =
      new FeatureGeneratorStage(
        extractFn = extractFn,
        extractSource = extractSource,
        aggregator = aggregator,
        outputName = name,
        outputIsResponse = isResponse,
        aggregateWindow = aggregateWindow
      )(tti, tto)

    originStage.getOutput().asInstanceOf[Feature[O]]
  }
}
