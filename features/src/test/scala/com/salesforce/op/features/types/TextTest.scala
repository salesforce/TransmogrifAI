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
