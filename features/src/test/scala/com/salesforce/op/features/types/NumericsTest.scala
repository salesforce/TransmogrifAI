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
class NumericsTest extends FlatSpec with TestCommon {

  /* Binary tests */
  Spec[Binary] should "extend OPNumeric" in {
    new Binary(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Binary(false).equals(new Binary(false)) shouldBe true
    new Binary(false).equals(new Binary(true)) shouldBe false
    new Binary(false).equals(new Binary(Option(false))) shouldBe true
    new Binary(false).equals(new Binary(Option(true))) shouldBe false
    new Binary(None).equals(new Binary(Option(true))) shouldBe false
    new Binary(None).equals(new Binary(Option(false))) shouldBe false
    new Binary(false).equals(new Real(5)) shouldBe false

    true.toBinary shouldBe a[Binary]
  }

  /* Real */
  Spec[Real] should "extend OPNumeric" in {
    new Real(None) shouldBe a[OPNumeric[_]]
  }
  it should "be able to convert to RealNN" in {
    new Real(5.0).toRealNN().value shouldBe Some(5.0)
    new Real(Option(5.0)).toRealNN().value shouldBe Some(5.0)
    new Real(None).toRealNN().value shouldBe Some(0.0)
    new Real(None).toRealNN(Option(5.0)).value shouldBe Some(5.0)
  }
  it should "compare values correctly" in {
    new Real(5.0).equals(new Real(5.0)) shouldBe true
    new Real(5.0).equals(new Real(1.0)) shouldBe false
    new Real(Option(5.0)).equals(new Binary(true)) shouldBe false

    3.14.toReal shouldBe a[Real]
    3.14.toRealNN shouldBe a[RealNN]
  }

  /* Integral */
  Spec[Integral] should "extend OPNumeric" in {
    new Integral(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Integral(5L).equals(new Integral(5L)) shouldBe true
    new Integral(Option(5L)).equals(new Binary(true)) shouldBe false

    849L.toIntegral shouldBe a[Integral]
  }

  /* Percent */
  Spec[Percent] should "extend Real" in {
    new Percent(None) shouldBe a[Real]
    new Percent(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Percent(5) == new Integral(5L) shouldBe true
    new Percent(2.0) == (new Binary(true)) shouldBe false

    3.4.toPercent shouldBe a[Percent]
  }

  /* Currency */
  Spec[Currency] should "extend Real" in {
    new Currency(None) shouldBe a[Real]
    new Currency(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Currency(5) == new Integral(5L) shouldBe true
    new Currency(2.0) == (new Binary(true)) shouldBe false

    4.50.toCurrency shouldBe a[Currency]
  }

  /* Date */
  Spec[Date] should "extend Integral" in {
    new Date(None) shouldBe a[Integral]
    new Date(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Date(5) == new Integral(5L) shouldBe true
    new Date(2) == (new Binary(true)) shouldBe false

    22L.toDate shouldBe a[Date]
  }

  /* DateTime */
  Spec[DateTime] should "extend Integral" in {
    new DateTime(None) shouldBe a[Date]
    new DateTime(None) shouldBe a[Integral]
    new DateTime(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new DateTime(5) == new Integral(5L) shouldBe true
    new DateTime(2) == (new Binary(true)) shouldBe false

    3402949L.toDateTime shouldBe a[DateTime]
  }
}
