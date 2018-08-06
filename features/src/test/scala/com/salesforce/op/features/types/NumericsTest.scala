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
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class NumericsTest extends FlatSpec with TestCommon {

  /* Binary tests */
  Spec[Binary] should "extend OPNumeric" in {
    new Binary(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Binary(false).equals(new Binary(false)) shouldBe true
    Binary(false).equals(new Binary(true)) shouldBe false
    Binary(false).equals(new Binary(Option(false))) shouldBe true
    new Binary(false).equals(new Binary(Option(true))) shouldBe false
    Binary(None).equals(new Binary(Option(true))) shouldBe false
    Binary(None).equals(Binary.empty) shouldBe true
    Binary(None).equals(new Binary(Option(false))) shouldBe false
    new Binary(false).equals(new Real(5)) shouldBe false

    true.toBinary shouldBe a[Binary]
  }
  it should "convert to Double correctly" in {
    true.toBinary.toDouble shouldBe Some(1.0)
    false.toBinary.toDouble shouldBe Some(0.0)
    Binary.empty.toDouble shouldBe None
  }

  /* Real */
  Spec[Real] should "extend OPNumeric" in {
    new Real(None) shouldBe a[OPNumeric[_]]
  }
  it should "be able to convert to RealNN" in {
    new Real(5.0).toRealNN(0.0).value shouldBe Some(5.0)
    new Real(Option(5.0)).toRealNN(0.0).value shouldBe Some(5.0)
    new Real(Option(5.0)).toRealNN(0.0).value shouldBe Some(5.0)
    Real(None).toRealNN(0.0).value shouldBe Some(0.0)
    new Real(None).toRealNN(5.0).value shouldBe Some(5.0)
    new Real(None) shouldBe Real.empty
  }
  it should "compare values correctly" in {
    new Real(5.0).equals(new Real(5.0)) shouldBe true
    new Real(5.0).equals(new Real(1.0)) shouldBe false
    new Real(Option(5.0)).equals(new Binary(true)) shouldBe false

    3.14.toReal shouldBe a[Real]
    3.14.toRealNN shouldBe a[RealNN]
  }
  it should "convert to Double correctly" in {
    1.0.toReal.toDouble shouldBe Some(1.0)
    Real.empty.toDouble shouldBe None
  }

  Spec[RealNN] should "extend OPNumeric" in {
    new RealNN(-1.0) shouldBe a[OPNumeric[_]]
  }
  it should "not allow empty values" in {
    intercept[NonNullableEmptyException](new RealNN(null))
    intercept[NonNullableEmptyException](new RealNN(None))
  }
  it should "compare values correctly" in {
    new RealNN(5L).equals(new Real(5L)) shouldBe true
    new RealNN(Some(5.0)).equals(new Real(Some(5.0))) shouldBe true
    RealNN(5.0).equals(Real(5.0)) shouldBe true

    849L.toRealNN shouldBe a[RealNN]
  }
  it should "convert to Double correctly" in {
    1.0.toRealNN.toDouble shouldBe Some(1.0)
  }

  /* Integral */
  Spec[Integral] should "extend OPNumeric" in {
    new Integral(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Integral(5L).equals(new Integral(5L)) shouldBe true
    new Integral(Option(5L)).equals(new Binary(true)) shouldBe false
    Integral(5L).equals(new Integral(5L)) shouldBe true
    Integral(Option(5L)).equals(new Binary(true)) shouldBe false
    new Integral(None).equals(Integral.empty) shouldBe true

    849L.toIntegral shouldBe a[Integral]
  }
  it should "convert to Double correctly" in {
    1L.toIntegral.toDouble shouldBe Some(1.0)
  }

  /* Percent */
  Spec[Percent] should "extend Real" in {
    new Percent(None) shouldBe a[Real]
    new Percent(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Percent(5) == new Integral(5L) shouldBe true
    new Percent(Some(5.0)) == new Integral(5L) shouldBe true
    new Percent(2.0) == new Binary(true) shouldBe false
    Percent(Some(5.0)) == new Integral(5L) shouldBe true
    Percent(2.0) == new Binary(true) shouldBe false
    Percent.empty == new Percent(None) shouldBe true

    3.4.toPercent shouldBe a[Percent]
  }
  it should "convert to Double correctly" in {
    1L.toPercent.toDouble shouldBe Some(1.0)
  }

  /* Currency */
  Spec[Currency] should "extend Real" in {
    new Currency(None) shouldBe a[Real]
    new Currency(1.0) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Currency(5) == new Integral(5L) shouldBe true
    new Currency(2.0) == new Binary(true) shouldBe false
    Currency(None) shouldBe a[Real]
    Currency(1.0) shouldBe a[Real]
    Currency(None) == Currency.empty shouldBe true

    4.50.toCurrency shouldBe a[Currency]
  }
  it should "convert to Double correctly" in {
    1L.toCurrency.toDouble shouldBe Some(1.0)
  }

  /* Date */
  Spec[Date] should "extend Integral" in {
    new Date(None) shouldBe a[Integral]
    new Date(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new Date(5) == new Integral(5L) shouldBe true
    new Date(2) == new Binary(true) shouldBe false
    Date(5) == new Integral(5L) shouldBe true
    Date(Some(2L)) == new Binary(true) shouldBe false
    Date.empty == Date.empty shouldBe true

    22L.toDate shouldBe a[Date]
  }
  it should "convert to Double correctly" in {
    1L.toDate.toDouble shouldBe Some(1.0)
  }

  /* DateTime */
  Spec[DateTime] should "extend Integral" in {
    new DateTime(None) shouldBe a[Date]
    new DateTime(None) shouldBe a[Integral]
    new DateTime(None) shouldBe a[OPNumeric[_]]
  }
  it should "compare values correctly" in {
    new DateTime(5) == new Integral(5L) shouldBe true
    new DateTime(2) == new Binary(true) shouldBe false
    DateTime(5) == new Integral(5L) shouldBe true
    DateTime(Some(2L)) == new Binary(true) shouldBe false
    DateTime(None) == DateTime.empty shouldBe true

    3402949L.toDateTime shouldBe a[DateTime]
  }
  it should "convert to Double correctly" in {
    1L.toDateTime.toDouble shouldBe Some(1.0)
  }

  Spec(classOf[Float]) should "convert to feature types" in {
    0.1f.toReal shouldBe Real(0.10000000149011612)
    0.1f.toRealNN shouldBe RealNN(0.10000000149011612)
    0.1f.toCurrency shouldBe Currency(0.10000000149011612)
    0.1f.toPercent shouldBe Percent(0.10000000149011612)
    5.1f.toBinary shouldBe Binary(true)
    0f.toBinary shouldBe Binary(false)
  }
  it should "convert to feature types for option" in {
    Option(0.1f).toReal shouldBe Real(0.10000000149011612)
    Option(0.1f).toRealNN(0f) shouldBe RealNN(0.10000000149011612)
    (None: Option[Float]).toRealNN(-1f) shouldBe RealNN(-1.0)
    Option(0.1f).toCurrency shouldBe Currency(0.10000000149011612)
    Option(0.1f).toPercent shouldBe Percent(0.10000000149011612)
    Option(5.1f).toBinary shouldBe Binary(true)
    Option(0f).toBinary shouldBe Binary(false)
  }

  Spec(classOf[java.lang.Float]) should "convert to feature types" in {
    new java.lang.Float(0.1f).toReal shouldBe Real(0.10000000149011612)
    new java.lang.Float(0.1f).toReal.toRealNN(0.0) shouldBe RealNN(0.10000000149011612)
    (null: java.lang.Float).toReal.toRealNN(-1.0) shouldBe RealNN(-1.0)
    new java.lang.Float(0.1f).toCurrency shouldBe Currency(0.10000000149011612)
    new java.lang.Float(0.1f).toPercent shouldBe Percent(0.10000000149011612)
    new java.lang.Float(5.1f).toDouble.toBinary shouldBe Binary(true)
    new java.lang.Float(0f).toDouble.toBinary shouldBe Binary(false)
  }
}
