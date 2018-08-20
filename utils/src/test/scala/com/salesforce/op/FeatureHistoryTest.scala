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

package com.salesforce.op

import com.salesforce.op.FeatureHistory.{OriginFeatureKey, StagesKey}
import com.salesforce.op.test.TestCommon
import org.apache.spark.sql.types.MetadataBuilder
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class FeatureHistoryTest extends FlatSpec with TestCommon {

  val feature1 = "feature1"
  val feature2 = "feature2"
  val stage1 = "stage1"
  val stage2 = "stage2"

  Spec[FeatureHistory] should "convert a feature history to metadata" in {
    val featureHistory = FeatureHistory(originFeatures = Seq(feature1, feature2), stages = Seq(stage1, stage2))

    val featureHistoryMetadata = featureHistory.toMetadata

    featureHistoryMetadata.contains(OriginFeatureKey) shouldBe true
    featureHistoryMetadata.contains(StagesKey) shouldBe true

    featureHistoryMetadata.getStringArray(OriginFeatureKey) shouldBe Array(feature1, feature2)
    featureHistoryMetadata.getStringArray(StagesKey) shouldBe Array(stage1, stage2)
  }

  it should "merge two instances" in {
    val featureHistory1 = FeatureHistory(originFeatures = Seq(feature1), stages = Seq(stage1))
    val featureHistory2 = FeatureHistory(originFeatures = Seq(feature2), stages = Seq(stage2))

    val featureHistoryCombined = featureHistory1.merge(featureHistory2)
    featureHistoryCombined.originFeatures shouldBe Seq(feature1, feature2)
    featureHistoryCombined.stages shouldBe Seq(stage1, stage2)
  }

  it should "create a metadata for a map" in {
    val featureHistory1 = FeatureHistory(originFeatures = Seq(feature1), stages = Seq(stage1))
    val featureHistory2 = FeatureHistory(originFeatures = Seq(feature2), stages = Seq(stage2))

    val map = Map(("1" -> featureHistory1), ("2" -> featureHistory2))
    val featureHistoryMetadata = FeatureHistory.toMetadata(map)

    featureHistoryMetadata.contains("1") shouldBe true
    featureHistoryMetadata.contains("2") shouldBe true

    val f1 = featureHistoryMetadata.getMetadata("1")

    f1.contains(OriginFeatureKey) shouldBe true
    f1.contains(StagesKey) shouldBe true

    f1.getStringArray(OriginFeatureKey) shouldBe Array(feature1)
    f1.getStringArray(StagesKey) shouldBe Array(stage1)

    val f2 = featureHistoryMetadata.getMetadata("2")

    f2.contains(OriginFeatureKey) shouldBe true
    f2.contains(StagesKey) shouldBe true

    f2.getStringArray(OriginFeatureKey) shouldBe Array(feature2)
    f2.getStringArray(StagesKey) shouldBe Array(stage2)
  }

  it should "create a map from metadata" in {

    val featureHistory1 = FeatureHistory(originFeatures = Seq(feature1), stages = Seq(stage1))
    val featureHistory2 = FeatureHistory(originFeatures = Seq(feature2), stages = Seq(stage2))

    val featureHistoryMapMetadata = new MetadataBuilder()
      .putMetadata("1", featureHistory1.toMetadata)
      .putMetadata("2", featureHistory2.toMetadata)
      .build()

    val featureHistoryMap = FeatureHistory.fromMetadataMap(featureHistoryMapMetadata)

    featureHistoryMap.contains("1") shouldBe true
    featureHistoryMap("1") shouldBe featureHistory1

    featureHistoryMap.contains("2") shouldBe true
    featureHistoryMap("2") shouldBe featureHistory2
  }
}

