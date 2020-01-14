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

import com.salesforce.op.test.TestCommon
import org.apache.spark.sql.types.MetadataBuilder
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class SensitiveFeatureInformationTest extends FlatSpec with TestCommon {

  val probName = 1.0
  val genderDetectResults: Seq[String] = Seq("ByIndex", "AnotherStrategy")
  val probMale = 0.25
  val probFemale = 0.50
  val probOther = 0.25
  val name = "feature"
  val mapKey: Option[String] = None
  val actionTaken = true

  val sensitiveFeatureInfo: SensitiveFeatureInformation.Name = SensitiveFeatureInformation.Name(
    probName, genderDetectResults, probMale, probFemale, probOther, name, mapKey, actionTaken
  )

  Spec[SensitiveFeatureInformation] should "convert sensitive feature information to metadata" in {
    val metadata = sensitiveFeatureInfo.toMetadata

    metadata.contains(SensitiveFeatureInformation.NameKey) shouldBe true
    metadata.contains(SensitiveFeatureInformation.MapKeyKey) shouldBe true
    metadata.contains(SensitiveFeatureInformation.ActionTakenKey) shouldBe true
    metadata.contains(SensitiveFeatureInformation.TypeKey) shouldBe true
    metadata.contains(SensitiveFeatureInformation.Name.ProbNameKey) shouldBe true
    metadata.contains(SensitiveFeatureInformation.Name.GenderDetectStratsKey) shouldBe true
    metadata.contains(SensitiveFeatureInformation.Name.ProbMaleKey) shouldBe true
    metadata.contains(SensitiveFeatureInformation.Name.ProbFemaleKey) shouldBe true
    metadata.contains(SensitiveFeatureInformation.Name.ProbOtherKey) shouldBe true

    metadata.getString(SensitiveFeatureInformation.NameKey) shouldBe name
    metadata.getString(SensitiveFeatureInformation.MapKeyKey) shouldBe ""
    metadata.getBoolean(SensitiveFeatureInformation.ActionTakenKey) shouldBe actionTaken
    metadata.getString(SensitiveFeatureInformation.TypeKey) shouldBe SensitiveFeatureInformation.Name.EntryName
    metadata.getDouble(SensitiveFeatureInformation.Name.ProbNameKey) shouldBe probName
    metadata.getStringArray(SensitiveFeatureInformation.Name.GenderDetectStratsKey) shouldBe genderDetectResults
    metadata.getDouble(SensitiveFeatureInformation.Name.ProbMaleKey) shouldBe probMale
    metadata.getDouble(SensitiveFeatureInformation.Name.ProbFemaleKey) shouldBe probFemale
    metadata.getDouble(SensitiveFeatureInformation.Name.ProbOtherKey) shouldBe probOther
  }

  it should "create metadata from a map" in {
    val info1 = sensitiveFeatureInfo
    val info2 = SensitiveFeatureInformation.Name(0.0, Seq(""), 0.0, 0.0, 0.0, "f2", Some("key"), actionTaken = true)
    val map = Map("1" -> Seq(info1), "2" -> Seq(info2))
    val metadata = SensitiveFeatureInformation.toMetadata(map)

    metadata.contains("1") shouldBe true
    metadata.contains("2") shouldBe true

    val f1 = metadata.getMetadataArray("1").head
    f1.contains(SensitiveFeatureInformation.NameKey) shouldBe true
    f1.contains(SensitiveFeatureInformation.MapKeyKey) shouldBe true
    f1.contains(SensitiveFeatureInformation.TypeKey) shouldBe true
    f1.contains(SensitiveFeatureInformation.TypeKey) shouldBe true
    f1.contains(SensitiveFeatureInformation.Name.GenderDetectStratsKey) shouldBe true
    f1.contains(SensitiveFeatureInformation.Name.ProbMaleKey) shouldBe true
    f1.contains(SensitiveFeatureInformation.Name.ProbFemaleKey) shouldBe true
    f1.contains(SensitiveFeatureInformation.Name.ProbOtherKey) shouldBe true
    f1.getStringArray(SensitiveFeatureInformation.Name.GenderDetectStratsKey) shouldBe genderDetectResults
    f1.getDouble(SensitiveFeatureInformation.Name.ProbMaleKey) shouldBe probMale
    f1.getDouble(SensitiveFeatureInformation.Name.ProbFemaleKey) shouldBe probFemale
    f1.getDouble(SensitiveFeatureInformation.Name.ProbOtherKey) shouldBe probOther

    val f2 = metadata.getMetadataArray("2").head
    f2.contains(SensitiveFeatureInformation.NameKey) shouldBe true
    f2.contains(SensitiveFeatureInformation.MapKeyKey) shouldBe true
    f2.contains(SensitiveFeatureInformation.TypeKey) shouldBe true
    f2.contains(SensitiveFeatureInformation.TypeKey) shouldBe true
    f2.contains(SensitiveFeatureInformation.Name.GenderDetectStratsKey) shouldBe true
    f2.contains(SensitiveFeatureInformation.Name.ProbMaleKey) shouldBe true
    f2.contains(SensitiveFeatureInformation.Name.ProbFemaleKey) shouldBe true
    f2.contains(SensitiveFeatureInformation.Name.ProbOtherKey) shouldBe true
    f2.getStringArray(SensitiveFeatureInformation.Name.GenderDetectStratsKey) shouldBe Seq("")
    f2.getDouble(SensitiveFeatureInformation.Name.ProbMaleKey) shouldBe 0.0
    f2.getDouble(SensitiveFeatureInformation.Name.ProbFemaleKey) shouldBe 0.0
    f2.getDouble(SensitiveFeatureInformation.Name.ProbOtherKey) shouldBe 0.0
  }

  it should "create a map from metadata" in {
    val info1 = sensitiveFeatureInfo
    val info2 = SensitiveFeatureInformation.Name(0.0, Seq(""), 0.0, 0.0, 0.0, "f2", Some("key"), actionTaken = true)

    val mapMetadata = new MetadataBuilder()
      .putMetadataArray("1", Array(info1.toMetadata))
      .putMetadataArray("2", Array(info2.toMetadata))
      .build()

    val map = SensitiveFeatureInformation.fromMetadataMap(mapMetadata)

    map.contains("1") shouldBe true
    map("1") shouldBe Array(info1)
    map.contains("2") shouldBe true
    map("2") shouldBe Array(info2)
  }
}

