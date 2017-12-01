/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
    println(PaloAlto ._1 + 1)
  }

  it should "behave on missing data" in {
    val sut = new Geolocation(List.empty[Double])
    sut.lat.isNaN shouldBe true
    sut.lon.isNaN shouldBe true
    sut.accuracy shouldBe GeolocationAccuracy.Unknown
  }

  it should "not accept missing longitude" in {
    assertThrows[IllegalArgumentException] {
      new Geolocation(List(PaloAlto._1))
    }
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
