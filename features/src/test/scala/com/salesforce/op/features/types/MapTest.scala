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
  Spec[TextMap] should "extend OPMap[_]" in {
    val myTextMap = new TextMap(Map.empty[String, String])
    myTextMap shouldBe a[FeatureType]
    myTextMap shouldBe a[OPCollection]
    myTextMap shouldBe a[OPMap[_]]
  }
  it should "compare values correctly" in {
    new TextMap(Map("Hello" -> "Bye")).equals(new TextMap(Map("Hello" -> "Bye"))) shouldBe true
    new TextMap(Map("Bye" -> "Hello")).equals(new TextMap(Map("Hello" -> "Bye"))) shouldBe false
    FeatureTypeDefaults.TextMap.equals(new TextMap(Map("Hello" -> "Bye"))) shouldBe false
    FeatureTypeDefaults.TextMap.equals(TextMap(Map.empty[String, String])) shouldBe true

    Map("Hello" -> "Good Day").toTextMap shouldBe a[TextMap]
  }

  /* EmailMap tests */
  Spec[EmailMap] should "extend OPMap[_], and have a working shortcut" in {
    val myEmailMap = new EmailMap(Map.empty[String, String])
    myEmailMap shouldBe a[FeatureType]
    myEmailMap shouldBe a[OPCollection]
    myEmailMap shouldBe a[OPMap[_]]

    Map("aaa" -> "bbb").toEmailMap shouldBe a[EmailMap]
  }

  /* Base64Map tests */
  Spec[Base64Map] should "extend OPMap[_], and have a working shortcut" in {
    val myBase64Map = new Base64Map(Map.empty[String, String])
    myBase64Map shouldBe a[FeatureType]
    myBase64Map shouldBe a[OPCollection]
    myBase64Map shouldBe a[OPMap[_]]

    Map("aaa" -> "bbb").toBase64Map shouldBe a[Base64Map]
  }

  /* PhoneMap tests */
  Spec[PhoneMap] should "extend OPMap[_], and have a working shortcut" in {
    val myPhoneMap = new PhoneMap(Map.empty[String, String])
    myPhoneMap shouldBe a[FeatureType]
    myPhoneMap shouldBe a[OPCollection]
    myPhoneMap shouldBe a[OPMap[_]]

    Map("aaa" -> "bbb").toPhoneMap shouldBe a[PhoneMap]
  }

  /* IDMap tests */
  Spec[IDMap] should "extend OPMap[_], and have a working shortcut" in {
    val myIDMap = new IDMap(Map.empty[String, String])
    myIDMap shouldBe a[FeatureType]
    myIDMap shouldBe a[OPCollection]
    myIDMap shouldBe a[OPMap[_]]

    Map("aaa" -> "bbb").toIDMap shouldBe a[IDMap]
  }

  /* URLMap tests */
  Spec[URLMap] should "extend OPMap[_], and have a working shortcut" in {
    val myURLMap = new URLMap(Map.empty[String, String])
    myURLMap shouldBe a[FeatureType]
    myURLMap shouldBe a[OPCollection]
    myURLMap shouldBe a[OPMap[_]]

    Map("aaa" -> "bbb").toURLMap shouldBe a[URLMap]
  }

  /* TextAreaMap tests */
  Spec[TextAreaMap] should "extend OPMap[_], and have a working shortcut" in {
    val myTextAreaMap = new TextAreaMap(Map.empty[String, String])
    myTextAreaMap shouldBe a[FeatureType]
    myTextAreaMap shouldBe a[OPCollection]
    myTextAreaMap shouldBe a[OPMap[_]]

    Map("aaa" -> "bbb").toTextAreaMap shouldBe a[TextAreaMap]
  }

  /* PickListMap tests */
  Spec[PickListMap] should "extend OPMap[_], and have a working shortcut" in {
    val myPickListMap = new PickListMap(Map.empty[String, String])
    myPickListMap shouldBe a[FeatureType]
    myPickListMap shouldBe a[OPCollection]
    myPickListMap shouldBe a[OPMap[_]]

    Map("aaa" -> "bbb").toPickListMap shouldBe a[PickListMap]
  }

  /* ComboBoxMap tests */
  Spec[ComboBoxMap] should "extend OPMap[_], and have a working shortcut" in {
    val myComboBoxMap = new ComboBoxMap(Map.empty[String, String])
    myComboBoxMap shouldBe a[FeatureType]
    myComboBoxMap shouldBe a[OPCollection]
    myComboBoxMap shouldBe a[OPMap[_]]

    Map("aaa" -> "bbb").toComboBoxMap shouldBe a[ComboBoxMap]
  }

  /* CountryMap tests */
  Spec[CountryMap] should "extend OPMap[_] and Location, and have a working shortcut" in {
    val myCountryMap = new CountryMap(Map.empty[String, String])
    myCountryMap shouldBe a[FeatureType]
    myCountryMap shouldBe a[OPCollection]
    myCountryMap shouldBe a[OPMap[_]]
    myCountryMap shouldBe a[Location]

    Map("Tuesday" -> "Bahrain").toCountryMap shouldBe a[CountryMap]
  }

  /* StateMap tests */
  Spec[StateMap] should "extend OPMap[_] and Location, and have a working shortcut" in {
    val myStateMap = new StateMap(Map.empty[String, String])
    myStateMap shouldBe a[FeatureType]
    myStateMap shouldBe a[OPCollection]
    myStateMap shouldBe a[OPMap[_]]
    myStateMap shouldBe a[Location]

    Map("Thursday" -> "British Columbia").toStateMap shouldBe a[StateMap]
  }

  /* CityMap tests */
  Spec[CityMap] should "extend OPMap[_] and Location, and have a working shortcut" in {
    val myCityMap = new CityMap(Map.empty[String, String])
    myCityMap shouldBe a[FeatureType]
    myCityMap shouldBe a[OPCollection]
    myCityMap shouldBe a[OPMap[_]]
    myCityMap shouldBe a[Location]

    Map("Wednesday" -> "Kyoto").toCityMap shouldBe a[CityMap]
  }

  /* PostalCodeMap tests */
  Spec[PostalCodeMap] should "extend OPMap[_] and Location, and have a working shortcut" in {
    val myPostalCodeMap = new PostalCodeMap(Map.empty[String, String])
    myPostalCodeMap shouldBe a[FeatureType]
    myPostalCodeMap shouldBe a[OPCollection]
    myPostalCodeMap shouldBe a[OPMap[_]]
    myPostalCodeMap shouldBe a[Location]

    Map("Wednesday" -> "78492").toPostalCodeMap shouldBe a[PostalCodeMap]
  }

  /* StreetMap tests */
  Spec[StreetMap] should "extend OPMap[_] and Location, and have a working shortcut" in {
    val myStreetMap = new StreetMap(Map.empty[String, String])
    myStreetMap shouldBe a[FeatureType]
    myStreetMap shouldBe a[OPCollection]
    myStreetMap shouldBe a[OPMap[_]]
    myStreetMap shouldBe a[Location]

    Map("Friday" -> "341 Evergreen Terrace").toStreetMap shouldBe a[StreetMap]
  }

  /* GeolocationMap tests */
  Spec[GeolocationMap] should "extend OPMap[Seq[Double]] and Location, and have a working shortcut" in {
    val myGeolocationMap = new GeolocationMap(Map.empty[String, Seq[Double]])
    myGeolocationMap shouldBe a[FeatureType]
    myGeolocationMap shouldBe a[OPCollection]
    myGeolocationMap shouldBe a[OPMap[_]]
    myGeolocationMap shouldBe a[Location]

    Map("Friday" -> Seq(-23.11, 45.20, 4.0)).toGeolocationMap shouldBe a[GeolocationMap]
  }

}
