/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks
import org.apache.spark.ml.linalg.Vector

import scala.collection.TraversableOnce
import scala.util.Try

@RunWith(classOf[JUnitRunner])
class FeatureTypeTest extends PropSpec with PropertyChecks with TestCommon {
  val emptyVals: Seq[FeatureType] = Seq(
    // Vector
    OPVector.empty,
    // Lists
    TextList.empty,
    DateList.empty,
    DateTimeList.empty,
    Geolocation.empty,
    // Maps
    Base64Map.empty,
    BinaryMap.empty,
    ComboBoxMap.empty,
    CurrencyMap.empty,
    DateMap.empty,
    DateTimeMap.empty,
    EmailMap.empty,
    IDMap.empty,
    IntegralMap.empty,
    MultiPickListMap.empty,
    PercentMap.empty,
    PhoneMap.empty,
    PickListMap.empty,
    RealMap.empty,
    TextAreaMap.empty,
    TextMap.empty,
    URLMap.empty,
    CountryMap.empty,
    StateMap.empty,
    CityMap.empty,
    PostalCodeMap.empty,
    StreetMap.empty,
    GeolocationMap.empty,
    // Numerics
    Binary.empty,
    Currency.empty,
    Date.empty,
    DateTime.empty,
    Integral.empty,
    Percent.empty,
    Real.empty,
    RealNN(0.0),
    // Sets
    MultiPickList.empty,
    // Text
    Base64.empty,
    ComboBox.empty,
    Email.empty,
    ID.empty,
    Phone.empty,
    PickList.empty,
    Text.empty,
    TextArea.empty,
    URL.empty,
    Country.empty,
    State.empty,
    City.empty,
    PostalCode.empty,
    Street.empty
  )
  val nonEmptyVals: Seq[FeatureType] = Seq(
    // Vector
    OPVector(Vectors.dense(0.0, 1.0, 2.0)),
    // Lists
    TextList(Seq("a", "b", "c")),
    DateList(Seq(0L, 1L, 100L)),
    DateTimeList(Seq(0L, 1L, 100L)),
    Geolocation((1.0, 1.0, 1.0)),
    // Maps
    Base64Map(Map("a" -> "b")),
    BinaryMap(Map("a" -> true)),
    ComboBoxMap(Map("a" -> "b")),
    CurrencyMap(Map("a" -> 1.0)),
    DateMap(Map("a" -> 1L)),
    DateTimeMap(Map("a" -> 1L)),
    EmailMap(Map("a" -> "a@b.com")),
    IDMap(Map("a" -> "abc")),
    IntegralMap(Map("a" -> 1L)),
    MultiPickListMap(Map("a" -> Set("b"))),
    PercentMap(Map("a" -> 1.0)),
    PhoneMap(Map("a" -> "555-1234567")),
    PickListMap(Map("a" -> "b")),
    RealMap(Map("a" -> 1.0)),
    TextAreaMap(Map("a" -> "b")),
    TextMap(Map("a" -> "b")),
    URLMap(Map("a" -> "http://www.salesforce.com")),
    CountryMap(Map("a" -> "USA")),
    StateMap(Map("a" -> "CA")),
    CityMap(Map("a" -> "Palo Alto")),
    PostalCodeMap(Map("a" -> "94071")),
    StreetMap(Map("a" -> "Emerson Ave")),
    GeolocationMap(Map("a" -> Seq(1.0, 1.0, 1.0))),
    Prediction(1.0, Array(0.0, 1.0), Array(2.0, 3.0)),
    // Numerics
    Binary(true),
    Currency(1.0),
    Date(1L),
    DateTime(1L),
    Integral(1L),
    Percent(1.0),
    Real(1.0),
    RealNN(1.0),
    // Sets
    MultiPickList(Set("b")),
    // Text
    Base64("a"),
    ComboBox("a"),
    Email("a"),
    ID("abc"),
    Phone("555-1234567"),
    PickList("a"),
    Text("a"),
    TextArea("a"),
    URL("http://www.salesforce.com"),
    Country("USA"),
    State("CA"),
    City("Palo Alto"),
    PostalCode("94071"),
    Street("Emerson Ave")
  )
  val featureTypesVals = Table("ft", emptyVals ++ nonEmptyVals: _*)

  property("is a feature type") {
    forAll(featureTypesVals) { ft => ft shouldBe a[FeatureType] }
  }
  property("is serializable") {
    forAll(featureTypesVals) { ft => ft shouldBe a[Serializable] }
  }
  property("value should not be null") {
    forAll(featureTypesVals) { ft => ft.value != null shouldBe true }
  }
  property("v should be a shortcut to value") {
    forAll(featureTypesVals) { ft => ft.value shouldBe ft.v  }
  }
  property("has value extractor methods") {
    forAll(featureTypesVals) {
      case ft@SomeValue(v) =>
        ft.isEmpty shouldBe false
        v shouldBe ft.value
      case ft =>
        ft.isEmpty shouldBe true
    }
  }
  property("should be only few non nullable types") {
    forAll(featureTypesVals) { ft =>
      whenever(!ft.isNullable) {
        ft.isInstanceOf[RealNN] || ft.isInstanceOf[Prediction] shouldBe true
        ft shouldBe a[NonNullable]
      }
    }
  }
  property("nonEmpty should equal to !isEmpty") {
    forAll(featureTypesVals) { ft => ft.nonEmpty shouldBe !ft.isEmpty }
  }
  property("value must be present when non empty") {
    forAll(featureTypesVals) { ft =>
      whenever(ft.nonEmpty) {
        ft match {
          case SomeValue(Some(v)) => v != null shouldBe true
          case SomeValue(v: TraversableOnce[_]) => v.size should be > 0
          case SomeValue(v: Vector) => v.size should be > 0
          case _ => fail("Feature value must not be empty")
        }
      }
    }
  }
  property("toString should return a valid string") {
    forAll(featureTypesVals) { ft =>
      val actual = ft.toString
      val v = ft.value match {
        case _ if ft.isEmpty => ""
        case Seq(lat: Double, lon: Double, acc: Double) if ft.isInstanceOf[Geolocation] =>
          f"$lat%.5f, $lon%.5f, ${GeolocationAccuracy.withValue(acc.toInt)}"
        case t: TraversableOnce[_] => t.mkString(", ")
        case x => x.toString
      }
      val expected = s"${ft.getClass.getSimpleName}($v)"

      actual shouldBe expected
    }
  }
  property("exists should apply predicate to value") {
    forAll(featureTypesVals) { ft =>
      if (ft.isEmpty) {
        ft.exists(_ => true) shouldBe false
        ft.exists(_ => false) shouldBe false
      }
      else {
        ft.exists(_ => true) shouldBe true
        ft.exists(_ => false) shouldBe false
      }
    }
  }
  property("contains should compare value against the specified one") {
    forAll(featureTypesVals) { ft =>
      ft.contains(ft.value) shouldBe ft.nonEmpty
      ft.contains(ft.value) shouldBe ft.exists(_ == ft.value)
    }
  }
  property("hashCode should equal to value hashCode") {
    forAll(featureTypesVals) { ft => ft.hashCode shouldBe ft.value.hashCode }
  }
  property("equal to self and not others") {
    forAll(featureTypesVals) { ft => {
      ft.equals(ft) shouldBe true
      ft == ft shouldBe true
      ft.equals(Binary(false)) shouldBe false
      ft == Binary(false) shouldBe false
    }}
  }
  property("return correct feature type tag") {
    forAll(featureTypesVals) { ft => {
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
