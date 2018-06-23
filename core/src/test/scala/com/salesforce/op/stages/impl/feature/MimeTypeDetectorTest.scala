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

package com.salesforce.op.stages.impl.feature

import java.io.FileInputStream

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.commons.io.IOUtils
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class MimeTypeDetectorTest extends OpTransformerSpec[Text, MimeTypeDetector] with Base64TestData {
  val inputData = randomData
  val transformer = new MimeTypeDetector().setInput(randomBase64)
  val expectedResult = expectedRandom

  it should "validate the type hint" in {
    assertThrows[IllegalArgumentException](new MimeTypeDetector().setTypeHint("blarg"))
  }
  it should "validate the ma bytes to parse" in {
    assertThrows[IllegalArgumentException](new MimeTypeDetector().setMaxBytesToParse(-1L))
  }
  it should "detect octet stream data" in {
    val mime = randomBase64.detectMimeTypes()
    mime.originStage shouldBe a[UnaryTransformer[_, _]]
    val result = mime.originStage.asInstanceOf[UnaryTransformer[Base64, Text]].transform(randomData)

    result.collect(mime) should contain theSameElementsInOrderAs expectedRandom
  }
  it should "detect other mime types" in {
    val mime = realBase64.detectMimeTypes()
    val result = mime.originStage.asInstanceOf[UnaryTransformer[Base64, Text]].transform(realData)

    result.collect(mime) should contain theSameElementsInOrderAs expectedMime
  }
  it should "detect other mime types with a json type hint" in {
    val mime = realBase64.detectMimeTypes(typeHint = Some("application/json"))
    val result = mime.originStage.asInstanceOf[UnaryTransformer[Base64, Text]].transform(realData)

    result.collect(mime) should contain theSameElementsInOrderAs expectedMimeJson
  }
}


trait Base64TestData {
  self: TestSparkContext =>

  lazy val (randomData, randomBase64) = TestFeatureBuilder(
    Base64.empty +: Base64("") +: RandomText.base64(0, 10000).take(10).toSeq
  )
  lazy val (realData, realBase64) = TestFeatureBuilder(
    Seq(
      "811harmo24to36.mp3", "820orig36to48.wav", "face.png",
      "log4j.properties", "note.xml", "RunnerParams.json",
      "dummy.csv", "Canon_40D.jpg", "sample.pdf"
    ).map(loadResourceAsBase64)
  )

  val expectedRandom = Text.empty +: Seq.fill(11)(Text("application/octet-stream"))

  val expectedMime = Seq(
    "audio/mpeg", "audio/vnd.wave", "image/png",
    "text/plain", "application/xml", "text/plain",
    "text/plain", "image/jpeg", "application/pdf"
  ).map(_.toText)

  val expectedMimeJson = Seq(
    "audio/mpeg", "audio/vnd.wave", "image/png",
    "application/json", "application/xml", "application/json",
    "application/json", "image/jpeg", "application/pdf"
  ).map(_.toText)

  def loadResourceAsBase64(name: String): Base64 = Base64 {
    val bytes = IOUtils.toByteArray(new FileInputStream(resourceFile(name = name)))
    new String(java.util.Base64.getEncoder.encode(bytes))
  }

}
