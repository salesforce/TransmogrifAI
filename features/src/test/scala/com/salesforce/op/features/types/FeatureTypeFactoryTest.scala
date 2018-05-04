/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.{Assertion, Matchers, PropSpec}
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.{PropertyChecks, TableFor1}

import scala.concurrent.duration._
import scala.util.{Failure, Success, Try}


@RunWith(classOf[JUnitRunner])
class FeatureTypeFactoryTest
  extends PropSpec with PropertyChecks with TestCommon with ConcurrentCheck with FeatureTypeAsserts {

  val featureTypeFactories: TableFor1[FeatureTypeFactory[_ <: FeatureType]] = Table("ft",
    // Vector
    FeatureTypeFactory[OPVector](),
    // Lists
    FeatureTypeFactory[TextList](),
    FeatureTypeFactory[DateList](),
    FeatureTypeFactory[DateTimeList](),
    FeatureTypeFactory[Geolocation](),
    // Maps
    FeatureTypeFactory[Base64Map](),
    FeatureTypeFactory[BinaryMap](),
    FeatureTypeFactory[ComboBoxMap](),
    FeatureTypeFactory[CurrencyMap](),
    FeatureTypeFactory[DateMap](),
    FeatureTypeFactory[DateTimeMap](),
    FeatureTypeFactory[EmailMap](),
    FeatureTypeFactory[IDMap](),
    FeatureTypeFactory[IntegralMap](),
    FeatureTypeFactory[MultiPickListMap](),
    FeatureTypeFactory[PercentMap](),
    FeatureTypeFactory[PhoneMap](),
    FeatureTypeFactory[PickListMap](),
    FeatureTypeFactory[RealMap](),
    FeatureTypeFactory[TextAreaMap](),
    FeatureTypeFactory[TextMap](),
    FeatureTypeFactory[URLMap](),
    FeatureTypeFactory[CountryMap](),
    FeatureTypeFactory[StateMap](),
    FeatureTypeFactory[CityMap](),
    FeatureTypeFactory[PostalCodeMap](),
    FeatureTypeFactory[StreetMap](),
    FeatureTypeFactory[GeolocationMap](),
    FeatureTypeFactory[Prediction](),
    // Numerics
    FeatureTypeFactory[Binary](),
    FeatureTypeFactory[Currency](),
    FeatureTypeFactory[Date](),
    FeatureTypeFactory[DateTime](),
    FeatureTypeFactory[Integral](),
    FeatureTypeFactory[Percent](),
    FeatureTypeFactory[Real](),
    FeatureTypeFactory[RealNN](),
    // Sets
    FeatureTypeFactory[MultiPickList](),
    // Text
    FeatureTypeFactory[Base64](),
    FeatureTypeFactory[ComboBox](),
    FeatureTypeFactory[Email](),
    FeatureTypeFactory[ID](),
    FeatureTypeFactory[Phone](),
    FeatureTypeFactory[PickList](),
    FeatureTypeFactory[Text](),
    FeatureTypeFactory[TextArea](),
    FeatureTypeFactory[URL](),
    FeatureTypeFactory[Country](),
    FeatureTypeFactory[State](),
    FeatureTypeFactory[City](),
    FeatureTypeFactory[PostalCode](),
    FeatureTypeFactory[Street]()
  )

  property("is a feature type factory") {
    forAll(featureTypeFactories) { ft => ft shouldBe a[FeatureTypeFactory[_]] }
  }
  property("is serializable") {
    forAll(featureTypeFactories) { ft => ft shouldBe a[Serializable] }
  }
  property("create a feature type instance of null") {
    forAll(featureTypeFactories)(ft => assertCreate(ft.newInstance(null)))
  }
  property("create a feature type instance in a timely fashion") {
    forAllConcurrentCheck[FeatureTypeFactory[_ <: FeatureType]](
      numThreads = 10, numInstancesPerThread = 50000, atMost = 10.seconds,
      table = featureTypeFactories,
      functionCheck = ft => assertCreate(ft.newInstance(null))
    )
  }
}

trait FeatureTypeAsserts {
  self: Matchers =>

  def assertCreate(ft: => FeatureType): Assertion = Try(ft) match {
    case Failure(e) =>
      e shouldBe a[NonNullableEmptyException]
    case Success(v) =>
      v should not be null
      v shouldBe a[FeatureType]
  }

}
