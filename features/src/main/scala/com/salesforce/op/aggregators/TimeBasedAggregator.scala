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
  compareFun: (Long, Long) => Boolean,
  val timeZero: Long
)(implicit val ttag: WeakTypeTag[T]) extends MonoidAggregator[Event[T], (Long, T#Value), T] {

  val ftFactory = FeatureTypeFactory[T]()

  val monoid: Monoid[(Long, T#Value)] = new Monoid[(Long, T#Value)] {
    val zero = timeZero -> FeatureTypeDefaults.default[T].value
    def plus(l: (Long, T#Value), r: (Long, T#Value)): (Long, T#Value) = if (compareFun(l._1, r._1)) r else l
  }

  def prepare(input: Event[T]): (Long, T#Value) = input.date -> input.value.v

  def present(reduction: (Long, T#Value)): T = ftFactory.newInstance(reduction._2)
}

/**
 * Gives last (most recent) value of feature
 * @param zeroValue zero for feature type
 * @param ttag feature type tag
 * @tparam T type of feature
 */
abstract class LastAggregator[T <: FeatureType](implicit ttag: WeakTypeTag[T]) extends
  TimeBasedAggregator(compareFun = (l: Long, r: Long) => l < r, timeZero = 0L)(ttag = ttag)


/**
 * Gives the first value of feature
 * @param zeroValue zero for feature type
 * @param ttag feature type tag
 * @tparam T type of feature
 */
abstract class FirstAggregator[T <: FeatureType](implicit ttag: WeakTypeTag[T]) extends
  TimeBasedAggregator(compareFun = (l: Long, r: Long) => l >= r, timeZero = Long.MaxValue)(ttag = ttag)


case object LastVector extends LastAggregator[OPVector]
case object FirstVector extends FirstAggregator[OPVector]

case object LastTextList extends LastAggregator[TextList]
case object FirstTextList extends FirstAggregator[TextList]

case object LastDateList extends LastAggregator[DateList]
case object FirstDateList extends FirstAggregator[DateList]

case object LastDateTimeList extends LastAggregator[DateTimeList]
case object FirstDateTimeList extends FirstAggregator[DateTimeList]

case object LastGeolocation extends LastAggregator[Geolocation]
case object FirstGeolocation extends FirstAggregator[Geolocation]

case object LastBase64Map extends LastAggregator[Base64Map]
case object FirstBase64Map extends FirstAggregator[Base64Map]

case object LastBinaryMap extends LastAggregator[BinaryMap]
case object FirstBinaryMap extends FirstAggregator[BinaryMap]

case object LastComboBoxMap extends LastAggregator[ComboBoxMap]
case object FirstComboBoxMap extends FirstAggregator[ComboBoxMap]

case object LastCurrencyMap extends LastAggregator[CurrencyMap]
case object FirstCurrencyMap extends FirstAggregator[CurrencyMap]

case object LastDateMap extends LastAggregator[DateMap]
case object FirstDateMap extends FirstAggregator[DateMap]

case object LastDateTimeMap extends LastAggregator[DateTimeMap]
case object FirstDateTimeMap extends FirstAggregator[DateTimeMap]

case object LastEmailMap extends LastAggregator[EmailMap]
case object FirstEmailMap extends FirstAggregator[EmailMap]

case object LastIDMap extends LastAggregator[IDMap]
case object FirstIDMap extends FirstAggregator[IDMap]

case object LastIntegralMap extends LastAggregator[IntegralMap]
case object FirstIntegralMap extends FirstAggregator[IntegralMap]

case object LastMultiPickListMap extends LastAggregator[MultiPickListMap]
case object FirstMultiPickListMap extends FirstAggregator[MultiPickListMap]

case object LastPercentMap extends LastAggregator[PercentMap]
case object FirstPercentMap extends FirstAggregator[PercentMap]

case object LastPhoneMap extends LastAggregator[PhoneMap]
case object FirstPhoneMap extends FirstAggregator[PhoneMap]

case object LastPickListMap extends LastAggregator[PickListMap]
case object FirstPickListMap extends FirstAggregator[PickListMap]

case object LastRealMap extends LastAggregator[RealMap]
case object FirstRealMap extends FirstAggregator[RealMap]

case object LastTextAreaMap extends LastAggregator[TextAreaMap]
case object FirstTextAreaMap extends FirstAggregator[TextAreaMap]

case object LastTextMap extends LastAggregator[TextMap]
case object FirstTextMap extends FirstAggregator[TextMap]

case object LastURLMap extends LastAggregator[URLMap]
case object FirstURLMap extends FirstAggregator[URLMap]

case object LastCountryMap extends LastAggregator[CountryMap]
case object FirstCountryMap extends FirstAggregator[CountryMap]

case object LastStateMap extends LastAggregator[StateMap]
case object FirstStateMap extends FirstAggregator[StateMap]

case object LastCityMap extends LastAggregator[CityMap]
case object FirstCityMap extends FirstAggregator[CityMap]

case object LastPostalCodeMap extends LastAggregator[PostalCodeMap]
case object FirstPostalCodeMap extends FirstAggregator[PostalCodeMap]

case object LastStreetMap extends LastAggregator[StreetMap]
case object FirstStreetMap extends FirstAggregator[StreetMap]

case object LastGeolocationMap extends LastAggregator[GeolocationMap]
case object FirstGeolocationMap extends FirstAggregator[GeolocationMap]

case object LastBinary extends LastAggregator[Binary]
case object FirstBinary extends FirstAggregator[Binary]

case object LastCurrency extends LastAggregator[Currency]
case object FirstCurrency extends FirstAggregator[Currency]

case object LastDate extends LastAggregator[Date]
case object FirstDate extends FirstAggregator[Date]

case object LastDateTime extends LastAggregator[DateTime]
case object FirstDateTime extends FirstAggregator[DateTime]

case object LastIntegral extends LastAggregator[Integral]
case object FirstIntegral extends FirstAggregator[Integral]

case object LastPercent extends LastAggregator[Percent]
case object FirstPercent extends FirstAggregator[Percent]

case object LastReal extends LastAggregator[Real]
case object FirstReal extends FirstAggregator[Real]

case object LastMultiPickList extends LastAggregator[MultiPickList]
case object FirstMultiPickList extends FirstAggregator[MultiPickList]

case object LastBase64 extends LastAggregator[Base64]
case object FirstBase64 extends FirstAggregator[Base64]

case object LastComboBox extends LastAggregator[ComboBox]
case object FirstComboBox extends FirstAggregator[ComboBox]

case object LastEmail extends LastAggregator[Email]
case object FirstEmail extends FirstAggregator[Email]

case object LastID extends LastAggregator[ID]
case object FirstID extends FirstAggregator[ID]

case object LastPhone extends LastAggregator[Phone]
case object FirstPhone extends FirstAggregator[Phone]

case object LastPickList extends LastAggregator[PickList]
case object FirstPickList extends FirstAggregator[Phone]

case object LastText extends LastAggregator[Text]
case object FirstText extends FirstAggregator[Text]

case object LastTextArea extends LastAggregator[TextArea]
case object FirstTextArea extends FirstAggregator[TextArea]

case object LastURL extends LastAggregator[URL]
case object FirstURL extends FirstAggregator[URL]

case object LastCountry extends LastAggregator[Country]
case object FirstCountry extends FirstAggregator[Country]

case object LastState extends LastAggregator[State]
case object FirstState extends FirstAggregator[State]

case object LastCity extends LastAggregator[City]
case object FirstCity extends FirstAggregator[City]

case object LastPostalCode extends LastAggregator[PostalCode]
case object FirstPostalCode extends FirstAggregator[PostalCode]

case object LastStreet extends LastAggregator[Street]
case object FirstStreet extends FirstAggregator[Street]



