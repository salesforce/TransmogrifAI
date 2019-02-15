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
import org.apache.lucene.spatial3d.geom.{GeoPoint, PlanetModel}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class GeolocationTest extends FlatSpec with TestCommon {
  val PaloAlto: (Double, Double) = (37.4419, -122.1430)

  Spec[Geolocation] should "extend OPList[Double]" in {
    val myGeolocation = new Geolocation(List.empty[Double])
    myGeolocation shouldBe a[FeatureType]
    myGeolocation shouldBe a[OPCollection]
    myGeolocation shouldBe a[OPList[_]]
  }

  it should "behave on missing data" in {
    val sut = new Geolocation(List.empty[Double])
    sut.lat.isNaN shouldBe true
    sut.lon.isNaN shouldBe true
    sut.accuracy shouldBe GeolocationAccuracy.Unknown
  }

  it should "not accept missing value" in {
    assertThrows[IllegalArgumentException](new Geolocation(List(PaloAlto._1)))
    assertThrows[IllegalArgumentException](new Geolocation(List(PaloAlto._1, PaloAlto._2)))
    assertThrows[IllegalArgumentException](new Geolocation((PaloAlto._1, PaloAlto._2, 123456.0)))
  }

  it should "compare values correctly" in {
    new Geolocation(List(32.399, 154.213, 6.0)).equals(new Geolocation(List(32.399, 154.213, 6.0))) shouldBe true
    new Geolocation(List(12.031, -23.44, 6.0)).equals(new Geolocation(List(32.399, 154.213, 6.0))) shouldBe false
    FeatureTypeDefaults.Geolocation.equals(new Geolocation(List(32.399, 154.213, 6.0))) shouldBe false
    FeatureTypeDefaults.Geolocation.equals(FeatureTypeDefaults.Geolocation) shouldBe true
    FeatureTypeDefaults.Geolocation.equals(Geolocation(List.empty[Double])) shouldBe true

    (35.123, -94.094, 5.0).toGeolocation shouldBe a[Geolocation]
  }

  it should "correctly generate a Lucene GeoPoint object" in {
    val myGeo = new Geolocation(List(32.399, 154.213, 6.0))
    myGeo.toGeoPoint shouldBe new GeoPoint(PlanetModel.WGS84, math.toRadians(myGeo.lat), math.toRadians(myGeo.lon))
  }

}
