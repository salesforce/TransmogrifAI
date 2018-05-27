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
