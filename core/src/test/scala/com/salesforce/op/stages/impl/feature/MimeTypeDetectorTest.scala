/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import java.io.FileInputStream

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.test.{PassengerSparkFixtureTest, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.commons.io.IOUtils
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class MimeTypeDetectorTest extends FlatSpec with PassengerSparkFixtureTest {

  val (randomData, f1) = TestFeatureBuilder(
    Base64.empty +: Base64("") +: RandomText.base64(0, 10000).take(10).toSeq
  )
  val (realData, f2) = TestFeatureBuilder(
    Seq(
      "811harmo24to36.mp3", "820orig36to48.wav", "face.png", "log4j.properties",
      "note.xml", "RunnerParams.json", "dummy.csv", "sample.pdf"
    ).map(loadResourceAsBase64)
  )

  Spec[MimeTypeDetector] should "detect octet stream data" in {
    val mime = f1.detectMimeTypes()
    val result = mime.originStage.asInstanceOf[UnaryTransformer[Base64, Text]].transform(randomData)

    result.collect(mime) should contain theSameElementsInOrderAs
      Text.empty +: Seq.fill(11)(Text("application/octet-stream"))
  }
  it should "detect other mime types" in {
    val mime = f2.detectMimeTypes()
    val result = mime.originStage.asInstanceOf[UnaryTransformer[Base64, Text]].transform(realData)

    result.collect(mime) should contain theSameElementsInOrderAs Seq(
      "audio/mpeg", "audio/vnd.wave", "image/png",
      "text/plain", "application/xml", "text/plain",
      "text/plain", "application/pdf"
    ).map(_.toText)
  }
  it should "detect other mime types with a json type hint" in {
    val mime = f2.detectMimeTypes(typeHint = Some("application/json"))
    val result = mime.originStage.asInstanceOf[UnaryTransformer[Base64, Text]].transform(realData)

    result.collect(mime) should contain theSameElementsInOrderAs Seq(
      "audio/mpeg", "audio/vnd.wave", "image/png",
      "application/json", "application/xml", "application/json",
      "application/json", "application/pdf"
    ).map(_.toText)
  }

  def loadResourceAsBase64(name: String): Base64 = Base64 {
    val bytes = IOUtils.toByteArray(new FileInputStream(resourceFile(name = name)))
    new String(java.util.Base64.getEncoder.encode(bytes))
  }

}
