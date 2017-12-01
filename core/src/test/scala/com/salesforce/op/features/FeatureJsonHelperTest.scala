/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features

import com.salesforce.op._
import com.salesforce.op.test.{PassengerFeaturesTest, TestCommon}
import org.json4s.MappingException
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class FeatureJsonHelperTest extends FlatSpec with PassengerFeaturesTest with TestCommon {

  trait DifferentParents {
    val feature = height + weight
    val stages = Map(feature.originStage.uid -> feature.originStage)
    val features = Map(height.uid -> height, weight.uid -> weight)
  }

  trait SameParents {
    val feature = height + height
    val stages = Map(feature.originStage.uid -> feature.originStage)
    val features = Map(height.uid -> height, height.uid -> height)
  }

  Spec(FeatureJsonHelper.getClass) should "serialize/deserialize a feature properly" in new DifferentParents {
    val json = feature.toJson()
    val parsedFeature = FeatureJsonHelper.fromJsonString(json, stages, features)
    if (parsedFeature.isFailure) fail(s"Failed to deserialize from json: $json", parsedFeature.failed.get)

    val res = parsedFeature.get
    res shouldBe a[Feature[_]]
    res.equals(feature) shouldBe true
    res.uid shouldBe feature.uid
    res.wtt.tpe =:= feature.wtt.tpe shouldBe true
  }

  it should "deserialize a set of parent features from one reference" in new SameParents {
    val json = feature.toJson()
    val parsedFeature = FeatureJsonHelper.fromJsonString(feature.toJson(), stages, features)
    if (parsedFeature.isFailure) fail(s"Failed to deserialize from json: $json", parsedFeature.failed.get)

    val res = parsedFeature.get
    res.equals(feature) shouldBe true
    res.wtt.tpe =:= feature.wtt.tpe shouldBe true
  }

  it should "fail to deserialize invalid json" in new DifferentParents {
    val res = FeatureJsonHelper.fromJsonString("{}", stages, features)
    res.isFailure shouldBe true
    res.failed.get shouldBe a[MappingException]
  }

  it should "fail when origin stage is not found" in new DifferentParents {
    val res = FeatureJsonHelper.fromJsonString(feature.toJson(), stages = Map.empty, features)
    res.isFailure shouldBe true
    res.failed.get shouldBe a[RuntimeException]
  }

  it should "fail when not all parents are found" in new DifferentParents {
    val res = FeatureJsonHelper.fromJsonString(feature.toJson(), stages, features = Map.empty)
    res.isFailure shouldBe true
    res.failed.get shouldBe a[RuntimeException]
  }


}
