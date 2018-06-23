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
      numThreads = 10, numInstancesPerThread = 50000, atMost = 10.seconds,
      table = featureTypeConverters,
      functionCheck = ft => {
        assertCreate(ft.fromSpark(null), (v: FeatureType) => {
          ft.asInstanceOf[FeatureTypeSparkConverter[FeatureType]].toSpark(v) shouldBe (null: Any)
        })
      }
    )
  }
}
