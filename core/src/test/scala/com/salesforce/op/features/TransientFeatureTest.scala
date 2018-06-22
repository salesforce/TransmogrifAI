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

package com.salesforce.op.features

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import com.salesforce.op.test.PassengerFeaturesTest
import com.salesforce.op.features.types._
import com.salesforce.op.test._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.{DefaultFormats, JObject}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Failure


@RunWith(classOf[JUnitRunner])
class TransientFeatureTest extends FlatSpec with PassengerFeaturesTest with TestCommon {

  implicit val formats = DefaultFormats
  val tf = TransientFeature(height)

  Spec[TransientFeature] should "return feature after contruction with a feature" in {
    compare(tf, height)
    tf.getFeature() shouldBe height
  }

  it should "contain the feature history information" in {
    val density = weight / height
    val tf = TransientFeature(density)
    compare(tf, density)
  }

  it should "construct properly without a feature" in {
    val t = TransientFeature(
      name = height.name,
      isResponse = height.isResponse,
      isRaw = height.isRaw,
      uid = height.uid,
      typeName = height.typeName,
      originFeatures = Seq(height.name),
      stages = Seq()
    )
    compare(t, height)
    assertThrows[RuntimeException] { t.getFeature() }
  }

  it should "be equal to self" in {
    tf shouldBe tf
    tf.equals(tf) shouldBe true
  }

  it should "not be equal to a different instance" in {
    val other = TransientFeature(weight)
    tf should not be other
    tf.equals(other) shouldBe false
  }

  it should "have hash code of it's uid" in {
    tf.hashCode() shouldBe tf.uid.hashCode
  }

  it should "cast back to FeatureLike" in {
    tf.asFeatureLike[Real] shouldBe height
  }

  it should "not serialize feature object" in {
    val tf2 = serdes(tf)
    compare(tf2, height)
    assertThrows[RuntimeException] { tf2.getFeature() }
    intercept[RuntimeException](tf2.asFeatureLike[Real])
  }

  it should "convert to json correctly" in {
    val jObj = tf.toJson
    (jObj \ "name").extract[String] shouldBe height.name
    (jObj \ "isResponse").extract[Boolean] shouldBe height.isResponse
    (jObj \ "isRaw").extract[Boolean] shouldBe height.isRaw
    (jObj \ "uid").extract[String] shouldBe height.uid
    (jObj \ "typeName").extract[String] shouldBe height.typeName.toString
    (jObj \ "originFeatures").extract[Seq[String]] should contain theSameElementsAs height.history().originFeatures
    (jObj \ "stages").extract[Seq[String]] should contain theSameElementsAs height.history().stages
  }

  it should "convert to json string correctly" in {
    val json = tf.toJsonString()
    val jObj = parse(json)
    (jObj \ "name").extract[String] shouldBe height.name
    (jObj \ "isResponse").extract[Boolean] shouldBe height.isResponse
    (jObj \ "isRaw").extract[Boolean] shouldBe height.isRaw
    (jObj \ "uid").extract[String] shouldBe height.uid
    (jObj \ "typeName").extract[String] shouldBe height.typeName.toString
    (jObj \ "originFeatures").extract[Seq[String]] should contain theSameElementsAs height.history().originFeatures
    (jObj \ "stages").extract[Seq[String]] should contain theSameElementsAs height.history().stages
  }

  it should "convert from json correctly" in {
    val tf2 = TransientFeature(tf.toJson)
    tf2.isSuccess shouldBe true
    compare(tf2.get, height)
    assertThrows[RuntimeException] { tf2.get.getFeature() }
  }

  it should "fail to convert from json if typeName is incorrect" in {
    val jObj =
      parse(
        """{"name": "test", "isResponse": true, "isRaw": true, "uid": "foo", "typeName": "fake",
          | "origins": [], "stages": []}""".stripMargin)
        .extract[JObject]
    val res = TransientFeature(jObj)
    res shouldBe a[Failure[_]]
    res.failed.get shouldBe a[ClassNotFoundException]
  }

  private def serdes[I](obj: I): I = {
    val bytes = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(bytes)
    oos.writeObject(obj)

    val ois = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray))
    ois.readObject().asInstanceOf[I]
  }

  private def compare(tf: TransientFeature, f: OPFeature): Unit = {
    tf.name shouldBe f.name
    tf.isRaw shouldBe f.isRaw
    tf.isResponse shouldBe f.isResponse
    tf.uid shouldBe f.uid
    tf.typeName shouldBe f.typeName
    tf.originFeatures shouldBe f.history().originFeatures
    tf.stages shouldBe f.history().stages
  }
}
