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

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.twitter.algebird.{Monoid, MonoidAggregator}

import scala.reflect.runtime.universe.WeakTypeTag

private[op] abstract class TimeBasedAggregator[T <: FeatureType]
(
  val compareFun: (Long, Long) => Boolean,
  val timeZero: Long,
  val emptyValue: T#Value
)(
  implicit val ttag: WeakTypeTag[T],
  val ttvag: WeakTypeTag[T#Value]
)
  extends MonoidAggregator[Event[T], (Long, T#Value), T] {
  val ftFactory = FeatureTypeFactory[T]()

  val monoid: Monoid[(Long, T#Value)] = new Monoid[(Long, T#Value)] {
    val zero = timeZero -> emptyValue
    def plus(l: (Long, T#Value), r: (Long, T#Value)): (Long, T#Value) = if (compareFun(l._1, r._1)) r else l
  }
  def prepare(input: Event[T]): (Long, T#Value) = input.date -> input.value.v

  def present(reduction: (Long, T#Value)): T = ftFactory.newInstance(reduction._2)
}

/**
 * Gives most recent value of feature
 * @param zeroValue zero for feature type
 * @param ttag feature type tag
 * @param ttvag feature value type tag
 * @tparam T type of feature
 */
abstract class MostRecentAggregator[T <: FeatureType]
(
  emptyValue: T#Value
)(
  implicit ttag: WeakTypeTag[T],
  ttvag: WeakTypeTag[T#Value]
) extends TimeBasedAggregator(
  compareFun = (l: Long, r: Long) => l < r, timeZero = 0L, emptyValue = emptyValue
)(ttag = ttag, ttvag = ttvag
)


/**
 * Gives the first value of feature
 * @param zeroValue zero for feature type
 * @param ttag feature type tag
 * @param ttvag feature value type tag
 * @tparam T type of feature
 */
abstract class FirstAggregator[T <: FeatureType]
(
  emptyValue: T#Value
)(
  implicit ttag: WeakTypeTag[T],
  ttvag: WeakTypeTag[T#Value]
) extends TimeBasedAggregator(
  compareFun = (l: Long, r: Long) => l >= r, timeZero = Long.MaxValue, emptyValue = emptyValue
)(ttag = ttag, ttvag = ttvag)


case object MostRecentVector extends MostRecentAggregator[OPVector](OPVector.empty.value)
case object FirstVector extends FirstAggregator[OPVector](OPVector.empty.value)

case object MostRecentTextList extends MostRecentAggregator[TextList](TextList.empty.value)
case object FirstTextList extends FirstAggregator[TextList](TextList.empty.value)

case object MostRecentDateList extends MostRecentAggregator[DateList](DateList.empty.value)
case object FirstDateList extends FirstAggregator[DateList](DateList.empty.value)

case object MostRecentDateTimeList extends MostRecentAggregator[DateTimeList](DateTimeList.empty.value)
case object FirstDateTimeList extends FirstAggregator[DateTimeList](DateTimeList.empty.value)

case object MostRecentGeolocation extends MostRecentAggregator[Geolocation](Geolocation.empty.value)
case object FirstGeolocation extends FirstAggregator[Geolocation](Geolocation.empty.value)

case object MostRecentBase64Map extends MostRecentAggregator[Base64Map](Base64Map.empty.value)
case object FirstBase64Map extends FirstAggregator[Base64Map](Base64Map.empty.value)

case object MostRecentBinaryMap extends MostRecentAggregator[BinaryMap](BinaryMap.empty.value)
case object FirstBinaryMap extends FirstAggregator[BinaryMap](BinaryMap.empty.value)

case object MostRecentComboBoxMap extends MostRecentAggregator[ComboBoxMap](ComboBoxMap.empty.value)
case object FirstComboBoxMap extends FirstAggregator[ComboBoxMap](ComboBoxMap.empty.value)

case object MostRecentCurrencyMap extends MostRecentAggregator[CurrencyMap](CurrencyMap.empty.value)
case object FirstCurrencyMap extends FirstAggregator[CurrencyMap](CurrencyMap.empty.value)

case object MostRecentDateMap extends MostRecentAggregator[DateMap](DateMap.empty.value)
case object FirstDateMap extends FirstAggregator[DateMap](DateMap.empty.value)

case object MostRecentDateTimeMap extends MostRecentAggregator[DateTimeMap](DateTimeMap.empty.value)
case object FirstDateTimeMap extends FirstAggregator[DateTimeMap](DateTimeMap.empty.value)

case object MostRecentEmailMap extends MostRecentAggregator[EmailMap](EmailMap.empty.value)
case object FirstEmailMap extends FirstAggregator[EmailMap](EmailMap.empty.value)

case object MostRecentIDMap extends MostRecentAggregator[IDMap](IDMap.empty.value)
case object FirstIDMap extends FirstAggregator[IDMap](IDMap.empty.value)

case object MostRecentIntegralMap extends MostRecentAggregator[IntegralMap](IntegralMap.empty.value)
case object FirstIntegralMap extends FirstAggregator[IntegralMap](IntegralMap.empty.value)

case object MostRecentMultiPickListMap extends MostRecentAggregator[MultiPickListMap](MultiPickListMap.empty.value)
case object FirstMultiPickListMap extends FirstAggregator[MultiPickListMap](MultiPickListMap.empty.value)

case object MostRecentPercentMap extends MostRecentAggregator[PercentMap](PercentMap.empty.value)
case object FirstPercentMap extends FirstAggregator[PercentMap](PercentMap.empty.value)

case object MostRecentPhoneMap extends MostRecentAggregator[PhoneMap](PhoneMap.empty.value)
case object FirstPhoneMap extends FirstAggregator[PhoneMap](PhoneMap.empty.value)

case object MostRecentPickListMap extends MostRecentAggregator[PickListMap](PickListMap.empty.value)
case object FirstPickListMap extends FirstAggregator[PickListMap](PickListMap.empty.value)

case object MostRecentRealMap extends MostRecentAggregator[RealMap](RealMap.empty.value)
case object FirstRealMap extends FirstAggregator[RealMap](RealMap.empty.value)

case object MostRecentTextAreaMap extends MostRecentAggregator[TextAreaMap](TextAreaMap.empty.value)
case object FirstTextAreaMap extends FirstAggregator[TextAreaMap](TextAreaMap.empty.value)

case object MostRecentTextMap extends MostRecentAggregator[TextMap](TextMap.empty.value)
case object FirstTextMap extends FirstAggregator[TextMap](TextMap.empty.value)

case object MostRecentURLMap extends MostRecentAggregator[URLMap](URLMap.empty.value)
case object FirstURLMap extends FirstAggregator[URLMap](URLMap.empty.value)

case object MostRecentCountryMap extends MostRecentAggregator[CountryMap](CountryMap.empty.value)
case object FirstCountryMap extends FirstAggregator[CountryMap](CountryMap.empty.value)

case object MostRecentStateMap extends MostRecentAggregator[StateMap](StateMap.empty.value)
case object FirstStateMap extends FirstAggregator[StateMap](StateMap.empty.value)

case object MostRecentCityMap extends MostRecentAggregator[CityMap](CityMap.empty.value)
case object FirstCityMap extends FirstAggregator[CityMap](CityMap.empty.value)

case object MostRecentPostalCodeMap extends MostRecentAggregator[PostalCodeMap](PostalCodeMap.empty.value)
case object FirstPostalCodeMap extends FirstAggregator[PostalCodeMap](PostalCodeMap.empty.value)

case object MostRecentStreetMap extends MostRecentAggregator[StreetMap](StreetMap.empty.value)
case object FirstStreetMap extends FirstAggregator[StreetMap](StreetMap.empty.value)

case object MostRecentGeolocationMap extends MostRecentAggregator[GeolocationMap](GeolocationMap.empty.value)
case object FirstGeolocationMap extends FirstAggregator[GeolocationMap](GeolocationMap.empty.value)

case object MostRecentBinary extends MostRecentAggregator[Binary](Binary.empty.value)
case object FirstBinary extends FirstAggregator[Binary](Binary.empty.value)

case object MostRecentCurrency extends MostRecentAggregator[Currency](Currency.empty.value)
case object FirstCurrency extends FirstAggregator[Currency](Currency.empty.value)

case object MostRecentDate extends MostRecentAggregator[Date](Date.empty.value)
case object FirstDate extends FirstAggregator[Date](Date.empty.value)

case object MostRecentDateTime extends MostRecentAggregator[DateTime](DateTime.empty.value)
case object FirstDateTime extends FirstAggregator[DateTime](DateTime.empty.value)

case object MostRecentIntegral extends MostRecentAggregator[Integral](Integral.empty.value)
case object FirstIntegral extends FirstAggregator[Integral](Integral.empty.value)

case object MostRecentPercent extends MostRecentAggregator[Percent](Percent.empty.value)
case object FirstPercent extends FirstAggregator[Percent](Percent.empty.value)

case object MostRecentReal extends MostRecentAggregator[Real](Real.empty.value)
case object FirstReal extends FirstAggregator[Real](Real.empty.value)

case object MostRecentRealNN extends MostRecentAggregator[RealNN](RealNN(0.0).value)
case object FirstRealNN extends FirstAggregator[RealNN](RealNN(0.0).value)

case object MostRecentMultiPickList extends MostRecentAggregator[MultiPickList](MultiPickList.empty.value)
case object FirstMultiPickList extends FirstAggregator[MultiPickList](MultiPickList.empty.value)

case object MostRecentBase64 extends MostRecentAggregator[Base64](Base64.empty.value)
case object FirstBase64 extends FirstAggregator[Base64](Base64.empty.value)

case object MostRecentComboBox extends MostRecentAggregator[ComboBox](ComboBox.empty.value)
case object FirstComboBox extends FirstAggregator[ComboBox](ComboBox.empty.value)

case object MostRecentEmail extends MostRecentAggregator[Email](Email.empty.value)
case object FirstEmail extends FirstAggregator[Email](Email.empty.value)

case object MostRecentID extends MostRecentAggregator[ID](ID.empty.value)
case object FirstID extends FirstAggregator[ID](ID.empty.value)

case object MostRecentPhone extends MostRecentAggregator[Phone](Phone.empty.value)
case object FirstPhone extends FirstAggregator[Phone](Phone.empty.value)

case object MostRecentPickList extends MostRecentAggregator[PickList](PickList.empty.value)
case object FirstPickList extends FirstAggregator[Phone](PickList.empty.value)

case object MostRecentText extends MostRecentAggregator[Text](Text.empty.value)
case object FirstText extends FirstAggregator[Text](Text.empty.value)

case object MostRecentTextArea extends MostRecentAggregator[TextArea](TextArea.empty.value)
case object FirstTextArea extends FirstAggregator[TextArea](TextArea.empty.value)

case object MostRecentURL extends MostRecentAggregator[URL](URL.empty.value)
case object FirstURL extends FirstAggregator[URL](URL.empty.value)

case object MostRecentCountry extends MostRecentAggregator[Country](Country.empty.value)
case object FirstCountry extends FirstAggregator[Country](Country.empty.value)

case object MostRecentState extends MostRecentAggregator[State](State.empty.value)
case object FirstState extends FirstAggregator[State](State.empty.value)

case object MostRecentCity extends MostRecentAggregator[City](City.empty.value)
case object FirstCity extends FirstAggregator[City](City.empty.value)

case object MostRecentPostalCode extends MostRecentAggregator[PostalCode](PostalCode.empty.value)
case object FirstPostalCode extends FirstAggregator[PostalCode](PostalCode.empty.value)

case object MostRecentStreet extends MostRecentAggregator[Street](Street.empty.value)
case object FirstStreet extends FirstAggregator[Street](Street.empty.value)



