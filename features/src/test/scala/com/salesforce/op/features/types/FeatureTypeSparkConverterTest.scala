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
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalacheck.Arbitrary._
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.{PropertyChecks, TableFor1}


import scala.collection.mutable.{WrappedArray => MWrappedArray}
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
  val strings = Gen.alphaNumStr

  val naturalNumbers = Gen.oneOf(
    arbitrary[Long], arbitrary[Int], arbitrary[Short], arbitrary[Byte]
  ).map(_.asInstanceOf[Number])

  val realNumbers = Gen.oneOf(arbitrary[Float], arbitrary[Double]).map(_.asInstanceOf[Number])

  val dateValues = Gen.oneOf(Gen.posNum[Int], Gen.posNum[Long]).map(_.asInstanceOf[Number])
  val dateTimeValues = Gen.posNum[Long]

  val booleans = Table("booleans", true, false)

  val featureTypeValues = Table("ft",
    Real.empty -> null,
    Text("abc") -> "abc",
    Real(0.1) -> 0.1,
    Integral(123) -> 123,
    Array(1.0, 2.0).toOPVector -> Vectors.dense(Array(1.0, 2.0)),
    Vectors.sparse(2, Array(0), Array(3.0)).toOPVector -> Vectors.sparse(2, Array(0), Array(3.0)),
    Seq(1L, 2L, 3L).toDateList -> Array(1L, 2L, 3L),
    Set("a", "b").toMultiPickList -> Array("a", "b")
  )

  val featureTypeMapValues = Table("ftm",
    TextMap.empty -> null,
    Map("1" -> 1.0, "2" -> 2.0).toRealMap -> Map("1" -> 1.0, "2" -> 2.0),
    Map("1" -> 3L, "2" -> 4L).toIntegralMap -> Map("1" -> 3L, "2" -> 4L),
    Map("1" -> "one", "2" -> "two").toTextMap -> Map("1" -> "one", "2" -> "two"),
    Map("1" -> Set("a", "b")).toMultiPickListMap -> Map("1" -> MWrappedArray.make(Array("a", "b"))),
    Map(NameStats.Keys.IsName.toString -> NameStats.BooleanStrings.True.toString,
      NameStats.Keys.Gender.toString -> NameStats.GenderStrings.Female.toString
    ).toNameStats -> Map(NameStats.Keys.IsName.toString -> NameStats.BooleanStrings.True.toString,
      NameStats.Keys.Gender.toString -> NameStats.GenderStrings.Female.toString
    ),
    Map("1" -> Seq(1.0, 5.0, 6.0)).toGeolocationMap -> Map("1" -> MWrappedArray.make(Array(1.0, 5.0, 6.0)))
  )

  val localMultiPickListValues = Table("mp",
    (Seq("a", "b"), new MultiPickList(Seq("a", "b").toSet)),
    (Nil, new MultiPickList(Set.empty[String])),
    (null, new MultiPickList(Set.empty[String]))
  )
  val localMultiPickListMapValues = Seq(Map("k0" -> Seq("a", "b", "c")))

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
    forAll(strings) { bogusName =>
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
  property("converts natural number of Byte/Short/Int/Long ranges to Integral feature type") {
    forAll(naturalNumbers) { nn =>
      FeatureTypeSparkConverter[Integral]().fromSpark(nn) shouldBe nn.longValue().toIntegral
      FeatureTypeSparkConverter.toSpark(nn.longValue().toIntegral) shouldEqual nn
    }
  }
  property("raises error on invalid natural numbers") {
    forAll(realNumbers)(nn =>
      intercept[IllegalArgumentException](FeatureTypeSparkConverter[Integral]().fromSpark(nn))
        .getMessage startsWith "Integral type mapping is not defined"
    )
  }
  property("converts real numbers in Float/Double ranges to Real feature type") {
    forAll(realNumbers) { rn =>
      FeatureTypeSparkConverter[Real]().fromSpark(rn) shouldBe rn.doubleValue().toReal
      FeatureTypeSparkConverter.toSpark(rn.doubleValue().toReal) shouldEqual rn
    }
  }
  property("converts natural numbers to Real feature type") {
    forAll(naturalNumbers) { rn =>
      FeatureTypeSparkConverter[Real]().fromSpark(rn) shouldBe rn.doubleValue().toReal
      FeatureTypeSparkConverter.toSpark(rn.doubleValue().toReal) shouldEqual rn
    }
  }
  property("converts natural numbers to RealNN feature type") {
    forAll(naturalNumbers) { rn =>
      FeatureTypeSparkConverter[RealNN]().fromSpark(rn) shouldBe rn.doubleValue().toReal
      FeatureTypeSparkConverter.toSpark(rn.doubleValue().toReal) shouldEqual rn
    }
  }
  property("raises error on invalid real numbers") {
    forAll(booleans) { b =>
      intercept[IllegalArgumentException](FeatureTypeSparkConverter[Real]().fromSpark(b))
        .getMessage startsWith "Real type mapping is not defined"
      intercept[IllegalArgumentException](FeatureTypeSparkConverter[RealNN]().fromSpark(b))
        .getMessage startsWith "RealNN type mapping is not defined"
    }
  }
  property("convert real numbers in Float/Double ranges to RealNN feature type") {
    forAll(realNumbers) { rn =>
      FeatureTypeSparkConverter[RealNN]().fromSpark(rn) shouldBe rn.doubleValue().toRealNN
      FeatureTypeSparkConverter.toSpark(rn.doubleValue().toRealNN) shouldEqual rn
    }
  }
  property("error for an empty RealNN value") {
    intercept[NonNullableEmptyException](FeatureTypeSparkConverter[RealNN]().fromSpark(null))
      .getMessage shouldBe "RealNN cannot be empty"
  }
  property("convert date denoted using Int/Long ranges to Date feature type") {
    forAll(dateValues) { dt =>
      FeatureTypeSparkConverter[Date]().fromSpark(dt) shouldBe dt.longValue().toDate
      FeatureTypeSparkConverter.toSpark(dt.longValue().toDate) shouldEqual dt
    }
  }
  property("error on invalid date values") {
    forAll(realNumbers)(rn =>
      intercept[IllegalArgumentException](FeatureTypeSparkConverter[Date]().fromSpark(rn))
        .getMessage startsWith "Date type mapping is not defined"
    )
  }
  property("convert timestamp denoted using Long range to Datetime feature type") {
    forAll(dateTimeValues) { dt =>
      FeatureTypeSparkConverter[DateTime]().fromSpark(dt) shouldBe dt.toDateTime
      FeatureTypeSparkConverter.toSpark(dt.toDateTime) shouldEqual dt
    }
  }
  property("convert string to text feature type") {
    forAll(strings) { s =>
      FeatureTypeSparkConverter[Text]().fromSpark(s) shouldBe s.toText
      FeatureTypeSparkConverter.toSpark(s.toText) shouldEqual s
    }
  }
  property("convert boolean to Binary feature type") {
    forAll(booleans) { b =>
      FeatureTypeSparkConverter[Binary]().fromSpark(b) shouldBe b.toBinary
      FeatureTypeSparkConverter.toSpark(b.toBinary) shouldEqual b
    }
  }
  property("convert feature type values to spark values") {
    forAll(featureTypeValues) { case (featureValue, sparkValue) =>
      FeatureTypeSparkConverter.toSpark(featureValue) shouldEqual sparkValue
    }
  }
  property("convert feature type map values to spark values") {
    forAll(featureTypeMapValues) { case (featureValue, sparkValue) =>
      FeatureTypeSparkConverter.toSpark(featureValue) shouldEqual sparkValue
    }
  }
  property("convert string seq to MultiPickList feature type") {
    forAll(localMultiPickListValues) { case (local, expected) =>
      FeatureTypeSparkConverter[MultiPickList]().fromSpark(local) shouldBe expected
    }
  }
  property("convert multi-picklist map to MultiPickListMap feature type") {
    localMultiPickListMapValues.foreach { local =>
      val expected = new MultiPickListMap(local.map{ case (k, v) => k -> v.toSet })
      FeatureTypeSparkConverter[MultiPickListMap]().fromSpark(local) shouldBe expected
    }
  }
}
