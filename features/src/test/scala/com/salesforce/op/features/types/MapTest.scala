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

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}
import scala.reflect.runtime.universe._


@RunWith(classOf[JUnitRunner])
class MapTest extends FlatSpec with TestCommon {

  /* TextMap tests */
  Spec[TextMap] should "extend correct base classes and have a working shortcut" in {
    val myTextMap = new TextMap(Map.empty[String, String])
    myTextMap shouldBe a[FeatureType]
    myTextMap shouldBe a[OPCollection]
    myTextMap shouldBe a[OPMap[_]]

    Map("aaa" -> "bbb").toTextMap shouldBe a[TextMap]
  }
  it should "compare values correctly" in {
    new TextMap(Map("Hello" -> "Bye")).equals(new TextMap(Map("Hello" -> "Bye"))) shouldBe true
    new TextMap(Map("Bye" -> "Hello")).equals(new TextMap(Map("Hello" -> "Bye"))) shouldBe false
    FeatureTypeDefaults.TextMap.equals(new TextMap(Map("Hello" -> "Bye"))) shouldBe false
    FeatureTypeDefaults.TextMap.equals(TextMap(Map.empty[String, String])) shouldBe true
  }

  /* EmailMap tests */
  Spec[EmailMap] should "extend correct base classes and have a working shortcut" in {
    val myEmailMap = new EmailMap(Map.empty[String, String])
    myEmailMap shouldBe a[FeatureType]
    myEmailMap shouldBe a[OPCollection]
    myEmailMap shouldBe a[OPMap[_]]
    myEmailMap shouldBe a[TextMap]

    Map("aaa" -> "bbb").toEmailMap shouldBe a[EmailMap]
  }

  /* Base64Map tests */
  Spec[Base64Map] should "extend correct base classes and have a working shortcut" in {
    val myBase64Map = new Base64Map(Map.empty[String, String])
    myBase64Map shouldBe a[FeatureType]
    myBase64Map shouldBe a[OPCollection]
    myBase64Map shouldBe a[OPMap[_]]
    myBase64Map shouldBe a[TextMap]

    Map("aaa" -> "bbb").toBase64Map shouldBe a[Base64Map]
  }

  /* PhoneMap tests */
  Spec[PhoneMap] should "extend correct base classes and have a working shortcut" in {
    val myPhoneMap = new PhoneMap(Map.empty[String, String])
    myPhoneMap shouldBe a[FeatureType]
    myPhoneMap shouldBe a[OPCollection]
    myPhoneMap shouldBe a[OPMap[_]]
    myPhoneMap shouldBe a[TextMap]

    Map("aaa" -> "bbb").toPhoneMap shouldBe a[PhoneMap]
  }

  /* IDMap tests */
  Spec[IDMap] should "extend correct base classes and have a working shortcut" in {
    val myIDMap = new IDMap(Map.empty[String, String])
    myIDMap shouldBe a[FeatureType]
    myIDMap shouldBe a[OPCollection]
    myIDMap shouldBe a[OPMap[_]]
    myIDMap shouldBe a[TextMap]

    Map("aaa" -> "bbb").toIDMap shouldBe a[IDMap]
  }

  /* URLMap tests */
  Spec[URLMap] should "extend correct base classes and have a working shortcut" in {
    val myURLMap = new URLMap(Map.empty[String, String])
    myURLMap shouldBe a[FeatureType]
    myURLMap shouldBe a[OPCollection]
    myURLMap shouldBe a[OPMap[_]]
    myURLMap shouldBe a[TextMap]

    Map("homepage" -> "http://www.salesforce.com").toURLMap shouldBe a[URLMap]
  }

  /* TextAreaMap tests */
  Spec[TextAreaMap] should "extend correct base classes and have a working shortcut" in {
    val myTextAreaMap = new TextAreaMap(Map.empty[String, String])
    myTextAreaMap shouldBe a[FeatureType]
    myTextAreaMap shouldBe a[OPCollection]
    myTextAreaMap shouldBe a[OPMap[_]]
    myTextAreaMap shouldBe a[TextMap]

    Map("aaa" -> "bbb").toTextAreaMap shouldBe a[TextAreaMap]
  }

  /* PickListMap tests */
  Spec[PickListMap] should "extend correct base classes and have a working shortcut" in {
    val myPickListMap = new PickListMap(Map.empty[String, String])
    myPickListMap shouldBe a[FeatureType]
    myPickListMap shouldBe a[OPCollection]
    myPickListMap shouldBe a[OPMap[_]]
    myPickListMap shouldBe a[TextMap]
    myPickListMap shouldBe a[SingleResponse]

    Map("aaa" -> "bbb").toPickListMap shouldBe a[PickListMap]
  }

  /* ComboBoxMap tests */
  Spec[ComboBoxMap] should "extend correct base classes and have a working shortcut" in {
    val myComboBoxMap = new ComboBoxMap(Map.empty[String, String])
    myComboBoxMap shouldBe a[FeatureType]
    myComboBoxMap shouldBe a[OPCollection]
    myComboBoxMap shouldBe a[OPMap[_]]
    myComboBoxMap shouldBe a[TextMap]

    Map("aaa" -> "bbb").toComboBoxMap shouldBe a[ComboBoxMap]
  }

  /* CountryMap tests */
  Spec[CountryMap] should "extend correct base classes and have a working shortcut" in {
    val myCountryMap = new CountryMap(Map.empty[String, String])
    myCountryMap shouldBe a[FeatureType]
    myCountryMap shouldBe a[OPCollection]
    myCountryMap shouldBe a[OPMap[_]]
    myCountryMap shouldBe a[TextMap]
    myCountryMap shouldBe a[Location]

    Map("Tuesday" -> "Bahrain").toCountryMap shouldBe a[CountryMap]
  }

  /* StateMap tests */
  Spec[StateMap] should "extend correct base classes and have a working shortcut" in {
    val myStateMap = new StateMap(Map.empty[String, String])
    myStateMap shouldBe a[FeatureType]
    myStateMap shouldBe a[OPCollection]
    myStateMap shouldBe a[OPMap[_]]
    myStateMap shouldBe a[TextMap]
    myStateMap shouldBe a[Location]

    Map("Thursday" -> "British Columbia").toStateMap shouldBe a[StateMap]
  }

  /* CityMap tests */
  Spec[CityMap] should "extend correct base classes and have a working shortcut" in {
    val myCityMap = new CityMap(Map.empty[String, String])
    myCityMap shouldBe a[FeatureType]
    myCityMap shouldBe a[OPCollection]
    myCityMap shouldBe a[OPMap[_]]
    myCityMap shouldBe a[TextMap]
    myCityMap shouldBe a[Location]

    Map("Wednesday" -> "Kyoto").toCityMap shouldBe a[CityMap]
  }

  /* PostalCodeMap tests */
  Spec[PostalCodeMap] should "extend correct base classes and have a working shortcut" in {
    val myPostalCodeMap = new PostalCodeMap(Map.empty[String, String])
    myPostalCodeMap shouldBe a[FeatureType]
    myPostalCodeMap shouldBe a[OPCollection]
    myPostalCodeMap shouldBe a[OPMap[_]]
    myPostalCodeMap shouldBe a[TextMap]
    myPostalCodeMap shouldBe a[Location]

    Map("Wednesday" -> "78492").toPostalCodeMap shouldBe a[PostalCodeMap]
  }

  /* StreetMap tests */
  Spec[StreetMap] should "extend correct base classes and have a working shortcut" in {
    val myStreetMap = new StreetMap(Map.empty[String, String])
    myStreetMap shouldBe a[FeatureType]
    myStreetMap shouldBe a[OPCollection]
    myStreetMap shouldBe a[OPMap[_]]
    myStreetMap shouldBe a[TextMap]
    myStreetMap shouldBe a[Location]

    Map("Friday" -> "341 Evergreen Terrace").toStreetMap shouldBe a[StreetMap]
  }

  /* GeolocationMap tests */
  Spec[GeolocationMap] should "extend correct base classes and have a working shortcut" in {
    val myGeolocationMap = new GeolocationMap(Map.empty[String, Seq[Double]])
    myGeolocationMap shouldBe a[FeatureType]
    myGeolocationMap shouldBe a[OPCollection]
    myGeolocationMap shouldBe a[OPMap[_]]
    myGeolocationMap shouldBe a[Location]

    Map("Friday" -> Seq(-23.11, 45.20, 4.0)).toGeolocationMap shouldBe a[GeolocationMap]
  }

  /* BinaryMao tests */
  Spec[BinaryMap] should "extend correct base classes and have a working shortcut" in {
    val binMap = new BinaryMap(Map.empty[String, Boolean])
    binMap shouldBe a[FeatureType]
    binMap shouldBe a[OPCollection]
    binMap shouldBe a[OPMap[_]]
    binMap shouldBe a[NumericMap]
    binMap shouldBe a[SingleResponse]

    Map("aaa" -> false).toBinaryMap shouldBe a[BinaryMap]
    Map("aaa" -> false, "bbb" -> true).toBinaryMap.toDoubleMap shouldBe Map("aaa" -> 0.0, "bbb" -> 1.0)
  }

  /* IntegralMap tests */
  Spec[IntegralMap] should "extend correct base classes and have a working shortcut" in {
    val intMap = new IntegralMap(Map.empty[String, Long])
    intMap shouldBe a[FeatureType]
    intMap shouldBe a[OPCollection]
    intMap shouldBe a[OPMap[_]]
    intMap shouldBe a[NumericMap]

    Map("aaa" -> 1L).toIntegralMap shouldBe a[IntegralMap]
    Map("aaa" -> 1L).toIntegralMap.toDoubleMap shouldBe Map("aaa" -> 1.0)
  }

  /* RealMap tests */
  Spec[RealMap] should "extend correct base classes and have a working shortcut" in {
    val rMap = new RealMap(Map.empty[String, Double])
    rMap shouldBe a[FeatureType]
    rMap shouldBe a[OPCollection]
    rMap shouldBe a[OPMap[_]]
    rMap shouldBe a[NumericMap]

    Map("aaa" -> 1.0).toRealMap shouldBe a[RealMap]
    Map("aaa" -> 2.0).toRealMap.toDoubleMap shouldBe Map("aaa" -> 2.0)
  }

  /* PercentMap tests */
  Spec[PercentMap] should "extend correct base classes and have a working shortcut" in {
    val pMap = new PercentMap(Map.empty[String, Double])
    pMap shouldBe a[FeatureType]
    pMap shouldBe a[OPCollection]
    pMap shouldBe a[OPMap[_]]
    pMap shouldBe a[RealMap]
    pMap shouldBe a[NumericMap]

    Map("aaa" -> 1.0).toPercentMap shouldBe a[PercentMap]
    Map("aaa" -> 2.0).toPercentMap.toDoubleMap shouldBe Map("aaa" -> 2.0)
  }

  /* CurrencyMap tests */
  Spec[CurrencyMap] should "extend correct base classes and have a working shortcut" in {
    val cMap = new CurrencyMap(Map.empty[String, Double])
    cMap shouldBe a[FeatureType]
    cMap shouldBe a[OPCollection]
    cMap shouldBe a[OPMap[_]]
    cMap shouldBe a[RealMap]
    cMap shouldBe a[NumericMap]

    Map("aaa" -> 1.0).toCurrencyMap shouldBe a[CurrencyMap]
    Map("aaa" -> 2.0).toCurrencyMap.toDoubleMap shouldBe Map("aaa" -> 2.0)
  }

  /* DateMap tests */
  Spec[DateMap] should "extend correct base classes and have a working shortcut" in {
    val m = new DateMap(Map.empty[String, Long])
    m shouldBe a[FeatureType]
    m shouldBe a[OPCollection]
    m shouldBe a[OPMap[_]]
    m shouldBe a[IntegralMap]
    m shouldBe a[NumericMap]

    Map("aaa" -> 1L).toDateMap shouldBe a[DateMap]
    Map("aaa" -> 2L).toDateMap.toDoubleMap shouldBe Map("aaa" -> 2.0)
  }

  /* DateTimeMap tests */
  Spec[DateTimeMap] should "extend correct base classes and have a working shortcut" in {
    val m = new DateTimeMap(Map.empty[String, Long])
    m shouldBe a[FeatureType]
    m shouldBe a[OPCollection]
    m shouldBe a[OPMap[_]]
    m shouldBe a[DateTimeMap]
    m shouldBe a[IntegralMap]
    m shouldBe a[NumericMap]

    Map("aaa" -> 1L).toDateTimeMap shouldBe a[DateTimeMap]
    Map("aaa" -> 2L).toDateTimeMap.toDoubleMap shouldBe Map("aaa" -> 2.0)
  }

  /* MultiPickListMap tests */
  Spec[MultiPickListMap] should "extend correct base classes and have a working shortcut" in {
    val m = new MultiPickListMap(Map.empty[String, Set[String]])
    m shouldBe a[FeatureType]
    m shouldBe a[OPCollection]
    m shouldBe a[OPMap[_]]
    m shouldBe a[MultiResponse]

    Map("aaa" -> Set("b")).toMultiPickListMap shouldBe a[MultiPickListMap]
  }

  /* NameStats tests */
  Spec[NameStats] should "extend correct base classes and have a working shortcut" in {
    val m = new NameStats(Map.empty[String, String])
    m shouldBe a[FeatureType]
    m shouldBe a[OPCollection]
    m shouldBe a[OPMap[_]]
    m shouldBe a[TextMap]

    Map(
      NameStats.Keys.IsNameIndicator -> NameStats.BooleanStrings.True,
      NameStats.Keys.Gender -> NameStats.GenderStrings.Female
    ).toNameStats shouldBe a[NameStats]
  }

  /* Prediction tests */
  Spec[Prediction] should "extend correct base classes" in {
    val m = Prediction(0.0)
    m shouldBe a[FeatureType]
    m shouldBe a[OPCollection]
    m shouldBe a[OPMap[_]]
    m shouldBe a[RealMap]

    Map("prediction" -> 1.0).toPrediction shouldBe a[Prediction]
    Map("prediction" -> 1.0).toPrediction.toDoubleMap shouldBe Map("prediction" -> 1.0)
  }

}
