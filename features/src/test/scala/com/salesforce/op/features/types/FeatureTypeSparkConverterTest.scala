/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.{PropertyChecks, TableFor1}

import scala.concurrent.duration._


@RunWith(classOf[JUnitRunner])
class FeatureTypeSparkConverterTest
  extends PropSpec with PropertyChecks with TestCommon with ConcurrentCheck with FeatureTypeAsserts {

  val featureTypeConverters: TableFor1[FeatureTypeSparkConverter[_ <: FeatureType]] = Table("ft",
    // Vector
    FeatureTypeSparkConverter[OPVector](),
    // Lists
    FeatureTypeSparkConverter[TextList](),
    FeatureTypeSparkConverter[DateList](),
    FeatureTypeSparkConverter[DateTimeList](),
    // Maps
    FeatureTypeSparkConverter[Base64Map](),
    FeatureTypeSparkConverter[BinaryMap](),
    FeatureTypeSparkConverter[ComboBoxMap](),
    FeatureTypeSparkConverter[CurrencyMap](),
    FeatureTypeSparkConverter[DateMap](),
    FeatureTypeSparkConverter[DateTimeMap](),
    FeatureTypeSparkConverter[EmailMap](),
    FeatureTypeSparkConverter[IDMap](),
    FeatureTypeSparkConverter[IntegralMap](),
    FeatureTypeSparkConverter[MultiPickListMap](),
    FeatureTypeSparkConverter[PercentMap](),
    FeatureTypeSparkConverter[PhoneMap](),
    FeatureTypeSparkConverter[PickListMap](),
    FeatureTypeSparkConverter[RealMap](),
    FeatureTypeSparkConverter[TextAreaMap](),
    FeatureTypeSparkConverter[TextMap](),
    FeatureTypeSparkConverter[URLMap](),
    FeatureTypeSparkConverter[CountryMap](),
    FeatureTypeSparkConverter[StateMap](),
    FeatureTypeSparkConverter[CityMap](),
    FeatureTypeSparkConverter[PostalCodeMap](),
    FeatureTypeSparkConverter[StreetMap](),
    FeatureTypeSparkConverter[GeolocationMap](),
    FeatureTypeSparkConverter[Prediction](),
    // Numerics
    FeatureTypeSparkConverter[Binary](),
    FeatureTypeSparkConverter[Currency](),
    FeatureTypeSparkConverter[Date](),
    FeatureTypeSparkConverter[DateTime](),
    FeatureTypeSparkConverter[Integral](),
    FeatureTypeSparkConverter[Percent](),
    FeatureTypeSparkConverter[Real](),
    FeatureTypeSparkConverter[RealNN](),
    // Sets
    FeatureTypeSparkConverter[MultiPickList](),
    // Text
    FeatureTypeSparkConverter[Base64](),
    FeatureTypeSparkConverter[ComboBox](),
    FeatureTypeSparkConverter[Email](),
    FeatureTypeSparkConverter[ID](),
    FeatureTypeSparkConverter[Phone](),
    FeatureTypeSparkConverter[PickList](),
    FeatureTypeSparkConverter[Text](),
    FeatureTypeSparkConverter[TextArea](),
    FeatureTypeSparkConverter[URL]()
  )

  property("is a feature type converter") {
    forAll(featureTypeConverters) { ft => ft shouldBe a[FeatureTypeSparkConverter[_]] }
  }
  property("is serializable") {
    forAll(featureTypeConverters) { ft => ft shouldBe a[Serializable] }
  }
  property("create a feature type instance of null") {
    forAll(featureTypeConverters)(ft => assertCreate(ft.fromSpark(null)))
  }
  property("create a feature type instance in a timely fashion") {
    forAllConcurrentCheck[FeatureTypeSparkConverter[_ <: FeatureType]](
      numThreads = 10, numInstancesPerThread = 50000, atMost = 10.seconds,
      table = featureTypeConverters,
      functionCheck = ft => assertCreate(ft.fromSpark(null))
    )
  }
}
