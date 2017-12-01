/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks

import scala.collection.TraversableOnce
import scala.util.Try

@RunWith(classOf[JUnitRunner])
class FeatureTypeTest extends PropSpec with PropertyChecks with TestCommon {

  val defaultFeatureTypes = Table("ft",
    // Vector
    FeatureTypeDefaults.OPVector,
    // Lists
    FeatureTypeDefaults.TextList,
    FeatureTypeDefaults.DateList,
    FeatureTypeDefaults.DateTimeList,
    FeatureTypeDefaults.Geolocation,
    // Maps
    FeatureTypeDefaults.Base64Map,
    FeatureTypeDefaults.BinaryMap,
    FeatureTypeDefaults.ComboBoxMap,
    FeatureTypeDefaults.CurrencyMap,
    FeatureTypeDefaults.DateMap,
    FeatureTypeDefaults.DateTimeMap,
    FeatureTypeDefaults.EmailMap,
    FeatureTypeDefaults.IDMap,
    FeatureTypeDefaults.IntegralMap,
    FeatureTypeDefaults.MultiPickListMap,
    FeatureTypeDefaults.PercentMap,
    FeatureTypeDefaults.PhoneMap,
    FeatureTypeDefaults.PickListMap,
    FeatureTypeDefaults.RealMap,
    FeatureTypeDefaults.TextAreaMap,
    FeatureTypeDefaults.TextMap,
    FeatureTypeDefaults.URLMap,
    FeatureTypeDefaults.CountryMap,
    FeatureTypeDefaults.StateMap,
    FeatureTypeDefaults.CityMap,
    FeatureTypeDefaults.PostalCodeMap,
    FeatureTypeDefaults.StreetMap,
    FeatureTypeDefaults.GeolocationMap,
    // Numerics
    FeatureTypeDefaults.Binary,
    FeatureTypeDefaults.Currency,
    FeatureTypeDefaults.Date,
    FeatureTypeDefaults.DateTime,
    FeatureTypeDefaults.Integral,
    FeatureTypeDefaults.Percent,
    FeatureTypeDefaults.Real,
    FeatureTypeDefaults.RealNN,
    // Sets
    FeatureTypeDefaults.MultiPickList,
    // Text
    FeatureTypeDefaults.Base64,
    FeatureTypeDefaults.ComboBox,
    FeatureTypeDefaults.Email,
    FeatureTypeDefaults.ID,
    FeatureTypeDefaults.Phone,
    FeatureTypeDefaults.PickList,
    FeatureTypeDefaults.Text,
    FeatureTypeDefaults.TextArea,
    FeatureTypeDefaults.URL,
    FeatureTypeDefaults.Country,
    FeatureTypeDefaults.State,
    FeatureTypeDefaults.City,
    FeatureTypeDefaults.PostalCode,
    FeatureTypeDefaults.Street
  )

  property("is a feature type") {
    forAll(defaultFeatureTypes) { ft => ft shouldBe a[FeatureType] }
  }

  property("is serializable") {
    forAll(defaultFeatureTypes) { ft => ft shouldBe a[Serializable] }
  }

  property("value should not be null") {
    forAll(defaultFeatureTypes) { ft => ft.value should not be null }
  }

  property("nullables should be empty") {
    forAll(defaultFeatureTypes) { ft =>
      whenever(ft.isNullable) {
        ft.isEmpty shouldBe true
      }
    }
  }

  property("nonEmpty should equal to !isEmpty") {
    forAll(defaultFeatureTypes) { ft => ft.nonEmpty shouldBe !ft.isEmpty }
  }

  property("toString should return a valid string") {
    forAll(defaultFeatureTypes) { ft =>
      val actual = ft.toString
      val v = ft.value match {
        case _ if ft.isEmpty => ""
        case t: TraversableOnce[_] => t.mkString(", ")
        case x => x.toString
      }
      val expected = s"${ft.getClass.getSimpleName}($v)"
      actual shouldBe expected
    }
  }

  property("hashCode should equal to value hashCode") {
    forAll(defaultFeatureTypes) { ft => ft.hashCode shouldBe ft.value.hashCode }
  }

  property("equal to self and not others") {
    forAll(defaultFeatureTypes) { ft => {
      ft.equals(ft) shouldBe true
      ft == ft shouldBe true
      ft.equals(Binary(true)) shouldBe false
      ft == Binary(true) shouldBe false
    }}
  }

  property("return correct feature type tag") {
    forAll(defaultFeatureTypes) { ft => {
      val ftt1 = Try(FeatureType.featureTypeTag(ft.getClass))
      if (ftt1.isFailure) fail(ftt1.failed.get)
      FeatureType.isFeatureType(ftt1.get) shouldBe true

      val ftt2 = Try(FeatureType.featureTypeTag(ft.getClass.getName))
      if (ftt2.isFailure) fail(ftt2.failed.get)
      FeatureType.isFeatureType(ftt2.get) shouldBe true

      ftt1.get.tpe =:= ftt2.get.tpe shouldBe true
    }}
  }

}
