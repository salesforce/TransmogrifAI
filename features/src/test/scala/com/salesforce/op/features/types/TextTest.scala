/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class TextTest extends FlatSpec with TestCommon {

  /* Text tests */
  Spec[Text] should "extend FeatureType" in {
    new Text(None) shouldBe a[FeatureType]
  }
  it should "compare values correctly" in {
    new Text("Hello").equals(new Text("Hello")) shouldBe true
    new Text("Hello").equals(new Text("Bye")) shouldBe false
    new Text(Some("Hello")).equals(new Text("Hello")) shouldBe true
    new Text(Some("Hello")).equals(new Text("Bye")) shouldBe false
    new Text(None).equals(new Text("Bye")) shouldBe false
    new Text(None).equals(new Text(None)) shouldBe true

    "Hello world".toText shouldBe a[Text]
  }

  /* Country tests */
  Spec[Country] should "extend Text and Location" in {
    new Country(None) shouldBe a[Text]
    new Country(None) shouldBe a[Location]

    "San Marino".toCountry shouldBe a[Country]
  }

  /* City tests */
  Spec[City] should "extend Text and Location" in {
    new City(None) shouldBe a[Text]
    new City(None) shouldBe a[Location]

    "Albuquerque".toCity shouldBe a[City]
  }

  /* PostalCode tests */
  Spec[PostalCode] should "extend Text and Location" in {
    new PostalCode(None) shouldBe a[Text]
    new PostalCode(None) shouldBe a[Location]

    "90210".toPostalCode shouldBe a[PostalCode]
  }

  /* State tests */
  Spec[State] should "extend Text and Location" in {
    new State(None) shouldBe a[Text]
    new State(None) shouldBe a[Location]

    "CA".toState shouldBe a[State]
  }

  /* Street tests */
  Spec[Street] should "extend Text and Location" in {
    new Street(None) shouldBe a[Text]
    new Street(None) shouldBe a[Location]

    "123 Fake St.".toStreet shouldBe a[Street]
  }

}
