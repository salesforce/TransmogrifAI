/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import org.apache.lucene.spatial3d.geom.{GeoPoint, PlanetModel}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class ListTest extends FlatSpec with TestCommon {

  /* TextList tests */
  Spec[TextList] should "extend OPList[String]" in {
    val myTextList = new TextList(List.empty[String])
    myTextList shouldBe a[FeatureType]
    myTextList shouldBe a[OPCollection]
    myTextList shouldBe a[OPList[_]]
  }
  it should "compare values correctly" in {
    new TextList(List("Hello", "Bye")).equals(new TextList(List("Hello", "Bye"))) shouldBe true
    new TextList(List("Bye", "Hello")).equals(new TextList(List("Hello", "Bye"))) shouldBe false
    FeatureTypeDefaults.TextList.equals(new TextList(List("Hello", "Bye"))) shouldBe false
    FeatureTypeDefaults.TextList.equals(TextList(List.empty[String])) shouldBe true

    List("Goodbye", "world").toTextList shouldBe a[TextList]
  }

  /* DateList tests */
  Spec[DateList] should "extend OPList[Long]" in {
    val myDateList = new DateList(List.empty[Long])
    myDateList shouldBe a[FeatureType]
    myDateList shouldBe a[OPCollection]
    myDateList shouldBe a[OPList[_]]
  }
  it should "compare values correctly" in {
    new DateList(List(456L, 13L)).equals(new DateList(List(456L, 13L))) shouldBe true
    new DateList(List(13L, 456L)).equals(new DateList(List(456L, 13L))) shouldBe false
    FeatureTypeDefaults.DateList.equals(new DateList(List(456L, 13L))) shouldBe false
    FeatureTypeDefaults.DateList.equals(DateList(List.empty[Long])) shouldBe true

    List(44829L, 389093L).toDateList shouldBe a[DateList]
  }

  /* DateTimeList tests */
  Spec[DateTimeList] should "extend OPList[Long]" in {
    val myDateTimeList = new DateTimeList(List.empty[Long])
    myDateTimeList shouldBe a[FeatureType]
    myDateTimeList shouldBe a[OPCollection]
    myDateTimeList shouldBe a[OPList[_]]
  }
  it should "compare values correctly" in {
    new DateTimeList(List(456L, 13L)).equals(new DateTimeList(List(456L, 13L))) shouldBe true
    new DateTimeList(List(13L, 456L)).equals(new DateTimeList(List(456L, 13L))) shouldBe false
    FeatureTypeDefaults.DateTimeList.equals(new DateTimeList(List(456L, 13L))) shouldBe false
    FeatureTypeDefaults.DateTimeList.equals(DateTimeList(List.empty[Long])) shouldBe true

    List(12237834L, 4890489839L).toDateTimeList shouldBe a[DateTimeList]
  }


}
