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

package com.salesforce.op.features

import com.salesforce.op._
import com.salesforce.op.filters.FeatureDistribution
import com.salesforce.op.test.{PassengerFeaturesTest, TestCommon}
import org.apache.spark.sql.types.MetadataBuilder
import org.json4s.MappingException
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class FeatureJsonHelperTest extends FlatSpec with PassengerFeaturesTest with TestCommon {

  trait DifferentParents {
    val feature = height + weight
    val stages = Map(
      feature.originStage.uid -> feature.originStage,
      height.originStage.uid -> height.originStage
    )
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

  it should "serialize and deserialize all information" in new DifferentParents {
    val dist = Seq(new FeatureDistribution(name = "dt", key = None, count = 1, nulls = 0,
    distribution = Array(0.5), summaryInfo = Array(1.0)))
    val meta = new MetadataBuilder().putString("test", "myValue").build()
    val withData = height.copy(distributions = dist, metadata = Option(meta))
    val jsonIn = withData.toJson()
    println(jsonIn)
    val parsedFeature = FeatureJsonHelper.fromJsonString(jsonIn, stages, features)
    println(parsedFeature)
    val res = parsedFeature.get
    res shouldBe a[Feature[_]]
    res.name shouldBe withData.name
    res.isResponse shouldBe withData.isResponse
    res.originStage shouldBe withData.originStage
    res.metadata shouldBe withData.metadata
    res.uid shouldBe withData.uid
    res.wtt.tpe =:= withData.wtt.tpe shouldBe true
  }


}
