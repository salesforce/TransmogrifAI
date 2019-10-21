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
import com.salesforce.op.testkit.RandomText
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class HumanNameIdentifierTest
  extends OpEstimatorSpec[NameStats, UnaryModel[Text, NameStats], UnaryEstimator[Text, NameStats]] {

  /**
   * Input Dataset to fit & transform
   */
  val (inputData, f1) = TestFeatureBuilder(Seq("NOTANAME").toText)

  /**
   * Estimator instance to be tested
   */
  val estimator: HumanNameIdentifier[Text] = new HumanNameIdentifier().setInput(f1)

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  val expectedResult: Seq[NameStats] = Seq(NameStats(Map.empty[String, String]))

  private def identifyName(data: Seq[Text]) = {
    val (newData, newFeature) = TestFeatureBuilder(data)
    val model = estimator.setInput(newFeature).fit(newData)
    val result: DataFrame = model.transform(newData)
    (newData, newFeature, model, result)
  }

  it should "identify a Text column with a single first name entry as Name" in {
    val (_, _, model, _) = identifyName(Seq("Robert").toText)
    model.asInstanceOf[HumanNameIdentifierModel[Text]].treatAsName shouldBe true
  }

  it should "not identify a Text column with a single non-name entry as Name" in {
    val (_, _, model, _) = identifyName(Seq("Firetruck").toText)
    model
      .asInstanceOf[HumanNameIdentifierModel[Text]]
      .treatAsName shouldBe false
  }

  it should "identify a Text column with multiple first name entries as Name" in {
    val names = RandomText.names.withProbabilityOfEmpty(0.0).take(100).toList
    val (_, _, model, _) = identifyName(names)
    model.asInstanceOf[HumanNameIdentifierModel[Text]].treatAsName shouldBe true
  }

  it should "identify a Text column with a single full name entry as Name" in {
    val (_, _, model, _) = identifyName(Seq("Elizabeth Warren").toText)
    model.asInstanceOf[HumanNameIdentifierModel[Text]].treatAsName shouldBe true
  }

  it should "not identify email addresses as Name" in {
    val (_, _, model, _) = identifyName(Seq("elizabeth@warren2020.com").toText)
    model
      .asInstanceOf[HumanNameIdentifierModel[Text]]
      .treatAsName shouldBe false
  }

  it should "not identify numbers as Name" in {
    val (_, _, model, _) =
      identifyName(Seq("1", "42", "0", "3000 michael").toText)
    model
      .asInstanceOf[HumanNameIdentifierModel[Text]]
      .treatAsName shouldBe false
  }

  it should "not identify a single repeated name as Name" in {
    val (_, _, model, _) = identifyName(Seq.fill(200)("Michael").toText)
    model
      .asInstanceOf[HumanNameIdentifierModel[Text]]
      .treatAsName shouldBe false
  }

  it should "return an empty map when the input is not a Name" in {
    val (_, _, _, result) = identifyName(Seq("Firetruck").toText)
    result.show()
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    // TODO: Figure out if there's a way to return an empty Map correctly instead of null
    // map.isEmpty shouldBe true
    map shouldBe null
  }

  it should "identify the gender of a single first Name correctly" in {
    import NameStats.GenderStrings._
    import NameStats.Keys._
    val (_, _, model, result) = identifyName(Seq("Alyssa").toText)
    model.asInstanceOf[HumanNameIdentifierModel[Text]].treatAsName shouldBe true
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    map.get(Gender) shouldBe Some(Female)
  }

  it should "identify which token is the first name in a single full name entry correctly" in {
    val (_, _, model, _) = identifyName(Seq("Shelby Bouvet").toText)
    model.asInstanceOf[HumanNameIdentifierModel[Text]].indexFirstName shouldBe 0
  }

  it should "identify the gender of a single full name entry correctly" in {
    import NameStats.GenderStrings._
    import NameStats.Keys._
    val (_, _, model, result) = identifyName(Seq("Shelby Bouvet").toText)
    model.asInstanceOf[HumanNameIdentifierModel[Text]].treatAsName shouldBe true
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    map.get(Gender) shouldBe Some(Female)
  }

  it should "identify the gender of a multiple full name entries (with varying token lengths) correctly" in {
    import NameStats.GenderStrings._
    import NameStats.Keys._
    val (_, _, model, result) = identifyName(Seq(
      "Sherrod Brown",
      "Maria Cantwell",
      "Benjamin L. Cardin",
      "Stephanie",
      "Thomas R. Carper"
    ).toText)
    model.asInstanceOf[HumanNameIdentifierModel[Text]].treatAsName shouldBe true
    val resultingMaps = result.collect().toSeq.map(row => row.get(1)).asInstanceOf[Seq[Map[String, String]]]
    val identifiedGenders = resultingMaps.map(_.get(Gender))
    identifiedGenders shouldBe Seq(Some(Male), Some(Female), Some(Male), Some(Female), Some(Male))
  }
}
