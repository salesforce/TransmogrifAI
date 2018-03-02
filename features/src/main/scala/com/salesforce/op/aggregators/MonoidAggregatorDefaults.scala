/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.twitter.algebird._

import scala.reflect.runtime.universe._

/**
 * Default monoid aggregators for each feature type
 */
object MonoidAggregatorDefaults {

  /**
   * Return a default monoid aggregator for the specified feature type
   *
   * Note: reflection lookups are VERY slow. use with caution!
   *
   * @tparam O feature type
   * @return a default monoid aggregator for the specified feature type
   */
  // TODO: do we need to support aggregators with different intermediate types?
  private[op] def aggregatorOf[O <: FeatureType : WeakTypeTag]: MonoidAggregator[Event[O], _, O] = {

    val aggregator = weakTypeOf[O] match {
      // Vector
      case wt if wt =:= weakTypeOf[OPVector] => UnionVector

      // Lists
      case wt if wt =:= weakTypeOf[TextList] => ConcatTextList
      case wt if wt =:= weakTypeOf[DateList] => ConcatDateList
      case wt if wt =:= weakTypeOf[DateTimeList] => ConcatDateTimeList
      case wt if wt =:= weakTypeOf[Geolocation] => GeolocationMidpoint

      // Maps
      case wt if wt =:= weakTypeOf[Base64Map] => UnionConcatBase64Map
      case wt if wt =:= weakTypeOf[BinaryMap] => UnionBinaryMap
      case wt if wt =:= weakTypeOf[ComboBoxMap] => UnionConcatComboBoxMap
      case wt if wt =:= weakTypeOf[CurrencyMap] => UnionCurrencyMap
      case wt if wt =:= weakTypeOf[DateMap] => UnionMaxDateMap
      case wt if wt =:= weakTypeOf[DateTimeMap] => UnionMaxDateTimeMap
      case wt if wt =:= weakTypeOf[EmailMap] => UnionConcatEmailMap
      case wt if wt =:= weakTypeOf[IDMap] => UnionConcatIDMap
      case wt if wt =:= weakTypeOf[IntegralMap] => UnionIntegralMap
      case wt if wt =:= weakTypeOf[MultiPickListMap] => UnionMultiPickListMap
      case wt if wt =:= weakTypeOf[PercentMap] => UnionMeanPercentMap
      case wt if wt =:= weakTypeOf[PhoneMap] => UnionConcatPhoneMap
      case wt if wt =:= weakTypeOf[PickListMap] => UnionConcatPickListMap
      case wt if wt =:= weakTypeOf[RealMap] => UnionRealMap
      case wt if wt =:= weakTypeOf[TextAreaMap] => UnionConcatTextAreaMap
      case wt if wt =:= weakTypeOf[TextMap] => UnionConcatTextMap
      case wt if wt =:= weakTypeOf[URLMap] => UnionConcatURLMap
      case wt if wt =:= weakTypeOf[CountryMap] => UnionConcatCountryMap
      case wt if wt =:= weakTypeOf[StateMap] => UnionConcatStateMap
      case wt if wt =:= weakTypeOf[CityMap] => UnionConcatCityMap
      case wt if wt =:= weakTypeOf[PostalCodeMap] => UnionConcatPostalCodeMap
      case wt if wt =:= weakTypeOf[StreetMap] => UnionConcatStreetMap
      case wt if wt =:= weakTypeOf[GeolocationMap] => UnionGeolocationMidpointMap

      // Numerics
      case wt if wt =:= weakTypeOf[Binary] => LogicalOr // TODO: reconsider using Xor since this is natural op on Z/2
      case wt if wt =:= weakTypeOf[Currency] => SumCurrency
      case wt if wt =:= weakTypeOf[Date] => MaxDate
      case wt if wt =:= weakTypeOf[DateTime] => MaxDateTime
      case wt if wt =:= weakTypeOf[Integral] => SumIntegral
      case wt if wt =:= weakTypeOf[Percent] => MeanPercent
      case wt if wt =:= weakTypeOf[Real] => SumReal
      case wt if wt =:= weakTypeOf[RealNN] => SumRealNN

      // Sets
      case wt if wt =:= weakTypeOf[MultiPickList] => UnionMultiPickList

      // Text
      case wt if wt =:= weakTypeOf[Base64] => ConcatBase64
      case wt if wt =:= weakTypeOf[ComboBox] => ConcatComboBox
      case wt if wt =:= weakTypeOf[Email] => ConcatEmail
      case wt if wt =:= weakTypeOf[ID] => ConcatID
      case wt if wt =:= weakTypeOf[Phone] => ConcatPhone
      case wt if wt =:= weakTypeOf[PickList] => ModePickList
      case wt if wt =:= weakTypeOf[Text] => ConcatText
      case wt if wt =:= weakTypeOf[TextArea] => ConcatTextArea
      case wt if wt =:= weakTypeOf[URL] => ConcatURL
      case wt if wt =:= weakTypeOf[Country] => ConcatCountry
      case wt if wt =:= weakTypeOf[State] => ConcatState
      case wt if wt =:= weakTypeOf[City] => ConcatCity
      case wt if wt =:= weakTypeOf[PostalCode] => ConcatPostalCode
      case wt if wt =:= weakTypeOf[Street] => ConcatStreet

      // Unknown
      case wt => throw new IllegalArgumentException(s"No default aggregator mapping for feature type $wt")
    }
    aggregator.asInstanceOf[MonoidAggregator[Event[O], _, O]]
  }

}
