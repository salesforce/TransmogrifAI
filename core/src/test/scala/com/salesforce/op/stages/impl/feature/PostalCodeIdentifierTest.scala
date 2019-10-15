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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class PostalCodeIdentifierTest
  extends OpEstimatorSpec[PostalCodeMap, UnaryModel[Text, PostalCodeMap], UnaryEstimator[Text, PostalCodeMap]] {
  /**
   * Input Dataset to fit & transform
   */
  val (inputData, f1) = TestFeatureBuilder(Seq("Michael").toText)

  /**
   * Estimator instance to be tested
   */
  val estimator: PostalCodeIdentifier[Text] = new PostalCodeIdentifier[Text]().setInput(f1)

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  val expectedResult: Seq[PostalCodeMap] = Seq(PostalCodeMap(Map.empty[String, String]))

  private def identifyPostalCode(data: Seq[Text]) = {
    val (newData, newFeature) = TestFeatureBuilder(data)
    val model = estimator.setInput(newFeature).fit(newData)
    val result: DataFrame = model.transform(newData)
    (newData, newFeature, model, result)
  }

  it should "identify a Text column with a single postal code as Postal Code" in {
    val (_, _, model, _) = identifyPostalCode(Seq("11581").toText)
    model.asInstanceOf[PostalCodeIdentifierModel[Text]].treatAsPostalCode shouldBe true
  }
  it should "identify postal codes as part of full addresses" in {
    val (_, _, model, result) = identifyPostalCode(Seq("1123 Birch Street, Amityville, NC, 72587").toText)
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    model.asInstanceOf[PostalCodeIdentifierModel[Text]].treatAsPostalCode shouldBe true
    map.get("postalCode") shouldBe Some("72587")
  }
  it should
    """identify postal codes as part of full addresses
      | even when there are other five digit numbers in the address""".stripMargin in {
    val (_, _, model, result) = identifyPostalCode(
      Seq("11235 Birch Street, Apartment 12345, Amityville, NC, 72587").toText)
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    model.asInstanceOf[PostalCodeIdentifierModel[Text]].treatAsPostalCode shouldBe true
    map.get("postalCode") shouldBe Some("72587")
  }
  it should "get the correct latitude and longitude" in {
    val (_, _, _, result) = identifyPostalCode(Seq("11581").toText)
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    map.get("lat") shouldBe Some("40.6523")
    map.get("lng") shouldBe Some("-73.7118")
  }
  it should "identify a Text column with a postal code sans one leading zero as Postal Code" in {
    val (_, _, model, result) = identifyPostalCode(Seq("8302").toText)
    model.asInstanceOf[PostalCodeIdentifierModel[Text]].treatAsPostalCode shouldBe true
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    map.get("lat") shouldBe Some("39.3762")
    map.get("lng") shouldBe Some("-75.1617")
  }
  it should "identify a Text column with a postal code sans two leading zeros as Postal Code" in {
    val (_, _, model, result) = identifyPostalCode(Seq("501").toText)
    model.asInstanceOf[PostalCodeIdentifierModel[Text]].treatAsPostalCode shouldBe true
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    map.get("lat") shouldBe Some("40.8154")
    map.get("lng") shouldBe Some("-73.0451")
  }
  it should "identify the five digit part of a nine digit postal code" in {
    val (_, _, model, result) = identifyPostalCode(Seq("1123 Birch Street, Amityville, NC, 72587-8302").toText)
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    model.asInstanceOf[PostalCodeIdentifierModel[Text]].treatAsPostalCode shouldBe true
    map.get("postalCode") shouldBe Some("72587")
  }
}
