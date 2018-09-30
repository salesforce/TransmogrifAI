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
import org.scalacheck.Gen
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.{PropertyChecks, TableFor1}

import scala.concurrent.duration._


@RunWith(classOf[JUnitRunner])
class FeatureTypeSparkConverterTest
  extends PropSpec with PropertyChecks with TestCommon with ConcurrentCheck with FeatureTypeAsserts {

  val featureTypeConverters: TableFor1[FeatureTypeSparkConverter[_ <: FeatureType]] = Table("ftc",
    FeatureTypeSparkConverter.featureTypeSparkConverters.values.toSeq: _*
  )
  val featureTypeNames: TableFor1[String] = Table("ftnames",
    FeatureTypeSparkConverter.featureTypeSparkConverters.keys.toSeq: _*
  )
  val bogusNames = Gen.alphaNumStr

  val naturalNumbers = Table("NaturalNumbers", Byte.MaxValue, Short.MaxValue, Int.MaxValue, Long.MaxValue)

  val realNumbers = Table("NaturalNumbers", Float.MaxValue, Double.MaxValue)

  val dateTimeValues = Table("DateTimeVals", 300, 300L)

  property("is a feature type converter") {
    forAll(featureTypeConverters) { ft => ft shouldBe a[FeatureTypeSparkConverter[_]] }
  }
  property("is serializable") {
    forAll(featureTypeConverters) { ft => ft shouldBe a[Serializable] }
  }
  property("make a converter by feature type name") {
    forAll(featureTypeNames) { featureTypeName =>
      val ft: FeatureTypeSparkConverter[_ <: FeatureType] =
        FeatureTypeSparkConverter.fromFeatureTypeName(featureTypeName)
      assertCreate(ft.fromSpark(null))
    }
  }
  property("error on making a converter on no existent feature type name") {
    forAll(bogusNames) { bogusName =>
      intercept[IllegalArgumentException](
        FeatureTypeSparkConverter.fromFeatureTypeName(bogusName)
      ).getMessage shouldBe s"Unknown feature type '$bogusName'"
    }
  }
  property("create a feature type instance of null") {
    forAll(featureTypeConverters)(ft => assertCreate(ft.fromSpark(null)))
  }
  property("create a feature type instance of null and back") {
    forAll(featureTypeConverters) { ft =>
      assertCreate(ft.fromSpark(null), (v: FeatureType) => {
        ft.asInstanceOf[FeatureTypeSparkConverter[FeatureType]].toSpark(v) shouldBe (null: Any)
        FeatureTypeSparkConverter.toSpark(v) shouldBe (null: Any)
      })
    }
  }
  property("create a feature type instance and back in a timely fashion") {
    forAllConcurrentCheck[FeatureTypeSparkConverter[_ <: FeatureType]](
      numThreads = 10, numInvocationsPerThread = 10000, atMost = 10.seconds,
      table = featureTypeConverters,
      functionCheck = ft => {
        assertCreate(ft.fromSpark(null), (v: FeatureType) => {
          ft.asInstanceOf[FeatureTypeSparkConverter[FeatureType]].toSpark(v) shouldBe (null: Any)
        })
      }
    )
  }

  property("converts Natural Number of Byte/Short/Int/Long ranges to Integral valued feature type") {
    forAll(naturalNumbers)(nn =>
      FeatureTypeSparkConverter[Integral]().fromSpark(nn) shouldBe a[Integral]
    )
  }
  property("converts Natural Number of Byte/Short/Int/Long ranges to Long range Integral feature") {
    forAll(naturalNumbers)(nn =>
      FeatureTypeSparkConverter[Integral]().fromSpark(nn).value.get shouldBe a[java.lang.Long]
    )
  }
  property("raises error for bad Natural Number") {
    forAll(realNumbers)(nn =>
      intercept[IllegalArgumentException](FeatureTypeSparkConverter[Integral]().fromSpark(nn)).getMessage
        shouldBe s"Integral type mapping is not defined for class java.lang.${nn.getClass.toString.capitalize}"
    )
  }

  property("converts Real Numbers in float/double ranges to Real valued feature type") {
    forAll(realNumbers)(rn =>
      FeatureTypeSparkConverter[Real]().fromSpark(rn) shouldBe a[Real]
    )
  }
  property("converts Real Numbers in float/double ranges to Double range Real feature") {
    forAll(realNumbers)(rn =>
      FeatureTypeSparkConverter[Real]().fromSpark(rn).value.get shouldBe a[java.lang.Double]
    )
  }
  property("raises error for bad Real Number") {
    forAll(naturalNumbers)(rn =>
      intercept[IllegalArgumentException](FeatureTypeSparkConverter[Real]().fromSpark(rn))
        .getMessage shouldBe s"Real type mapping is not defined for class java.lang.${rn.getClass.toString.capitalize}"
    )
  }

  property("converts Real Numbers in float/double ranges to RealNN valued feature type") {
    forAll(realNumbers)(rn =>
      FeatureTypeSparkConverter[RealNN]().fromSpark(rn) shouldBe a[RealNN]
    )
  }
  property("converts Real Numbers in float/double ranges Double range RealNN feature") {
    forAll(realNumbers)(rn =>
      FeatureTypeSparkConverter[RealNN]().fromSpark(rn).value.get shouldBe a[java.lang.Double]
    )
  }
  property("raises error for empty RealNN Number") {
    forAll(naturalNumbers)(rn =>
      intercept[NonNullableEmptyException](FeatureTypeSparkConverter[RealNN]().fromSpark(null))
        .getMessage shouldBe "RealNN cannot be empty"
    )
  }

  property("converts date denoted using int/long ranges to date feature types") {
    forAll(dateTimeValues)(dt =>
      FeatureTypeSparkConverter[Date]().fromSpark(dt) shouldBe a[Date]
    )
  }
  property("converts date denoted using int/long ranges to Long range date feature") {
    forAll(dateTimeValues)(dt =>
      FeatureTypeSparkConverter[Date]().fromSpark(dt).value.get shouldBe a[java.lang.Long]
    )
  }
  property("raises error for bad date values") {
    forAll(realNumbers)(rn =>
    intercept[IllegalArgumentException](FeatureTypeSparkConverter[Date]().fromSpark(rn))
      .getMessage shouldBe s"Date type mapping is not defined for class java.lang.${rn.getClass.toString.capitalize}"
    )
  }

  property("converts timestamp denoted using long range to datetime feature type") {
    forAll(dateTimeValues)(dt =>
      FeatureTypeSparkConverter[DateTime]().fromSpark(dt) shouldBe a[DateTime]
    )
  }
  property("converts timestamp denoted using long range to date super feature type") {
    forAll(dateTimeValues)(dt =>
      FeatureTypeSparkConverter[DateTime]().fromSpark(dt) shouldBe a[Date]
    )
  }
  property("converts timestamp denoted using long ranges to long range datetime feature") {
    forAll(dateTimeValues)(dt =>
      FeatureTypeSparkConverter[DateTime]().fromSpark(dt).value.get shouldBe a[java.lang.Long]
    )
  }

  property("converts string to text feature type") {
    val text = FeatureTypeSparkConverter[Text]().fromSpark("Simple")
    text shouldBe a[Text]
    text.value.get shouldBe a[String]
  }

  property("converts a Boolean to Binary feature type") {
    val bool = FeatureTypeSparkConverter[Binary]().fromSpark(false)
    bool shouldBe a[Binary]
    bool.value.get shouldBe a[java.lang.Boolean]
  }
}
