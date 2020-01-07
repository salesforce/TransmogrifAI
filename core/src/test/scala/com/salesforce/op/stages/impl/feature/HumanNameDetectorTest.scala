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

import com.salesforce.op.features.types.NameStats.GenderStrings._
import com.salesforce.op.features.types.NameStats.Keys._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.stages.NameDetectUtils
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.Metadata
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class HumanNameDetectorTest
  extends OpEstimatorSpec[NameStats, UnaryModel[Text, NameStats], UnaryEstimator[Text, NameStats]] {

  /**
   * Input Dataset to fit & transform
   */
  val (inputData, f1) = TestFeatureBuilder(Seq("NOTANAME").toText)

  /**
   * Estimator instance to be tested
   */
  val estimator: HumanNameDetector[Text] = new HumanNameDetector().setInput(f1)

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  val expectedResult: Seq[NameStats] = Seq(NameStats(Map.empty[String, String]))

  private lazy val NameDictionaryGroundTruth: RandomText[Text] = RandomText.textFromDomain(
    NameDetectUtils.DefaultNameDictionary.toList
  )

  private def identifyName(data: Seq[Text]) = {
    val (newData, newFeature) = TestFeatureBuilder(data)
    val model = estimator.setInput(newFeature).fit(newData)
    val result: DataFrame = model.transform(newData)
    (newData, newFeature, model, result)
  }

  it should "identify a Text column with a single first name entry as Name" in {
    val (_, _, model, _) = identifyName(Seq("Robert").toText)
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
  }

  it should "not identify a Text column with a single non-name entry as Name" in {
    val (_, _, model, _) = identifyName(Seq("Firetruck").toText)
    model
      .asInstanceOf[HumanNameDetectorModel[Text]]
      .treatAsName shouldBe false
  }

  it should "identify a Text column with multiple first name entries as Name" in {
    val names = NameDictionaryGroundTruth.withProbabilityOfEmpty(0.0).take(100).toList
    val (_, _, model, _) = identifyName(names)
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
  }

  it should "detect names based on the threshold correctly" in {
    val N = 50
    for {i <- Seq(2, 6)} {
      val numberNames = (N / 10) * i
      val names =
        NameDictionaryGroundTruth.withProbabilityOfEmpty(0.0).take(numberNames).toList ++
        RandomText.phones.withProbabilityOfEmpty(0.0).take(N - numberNames).toList.map(_.toString.toText)
      val (newData, newFeature) = TestFeatureBuilder(names)
      val newEstimator = new HumanNameDetector().setInput(newFeature)

      val threshold = numberNames.toDouble / N
      val modelBelowThreshold = newEstimator.setThreshold(threshold - 0.09).fit(newData)
      val modelAboveThreshold = newEstimator.setThreshold(threshold + 0.09).fit(newData)
      modelBelowThreshold.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
      modelAboveThreshold.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe false
    }
  }

  it should "identify a Text column with a single full name entry as Name" in {
    val (_, _, model, _) = identifyName(Seq("Elizabeth Warren").toText)
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
  }

  it should "not identify email addresses as Name" in {
    val (_, _, model, _) = identifyName(Seq("elizabeth@warren2020.com").toText)
    model
      .asInstanceOf[HumanNameDetectorModel[Text]]
      .treatAsName shouldBe false
  }

  it should "not identify numbers as Name" in {
    val (_, _, model, _) =
      identifyName(Seq("1", "42", "0", "3000 michael").toText)
    model
      .asInstanceOf[HumanNameDetectorModel[Text]]
      .treatAsName shouldBe false
  }

  it should "not identify a single repeated name as Name" in {
    val (_, _, model, _) = identifyName(Seq.fill(200)("Michael").toText)
    model
      .asInstanceOf[HumanNameDetectorModel[Text]]
      .treatAsName shouldBe false
  }

  it should "return an empty map when the input is not a Name" in {
    val (_, _, _, result) = identifyName(Seq("Firetruck").toText)
    result.show()
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    map shouldBe null
  }

  it should "identify the gender of a single first Name correctly" in {
    val (_, _, model, result) = identifyName(Seq("Alyssa").toText)
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    map.get(Gender) shouldBe Some(Female)
  }

  it should "identify which token is the first name in a single full name entry correctly" in {
    val (_, _, model, _) = identifyName(Seq("Shelby Bouvet").toText)
    model.asInstanceOf[HumanNameDetectorModel[Text]].orderedGenderDetectStrategies.head shouldBe
      GenderDetectStrategy.ByIndex(0)
  }

  it should "identify the gender of a single full name entry correctly" in {
    val (_, _, model, result) = identifyName(Seq("Shelby Bouvet").toText)
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
    val map = result.collect().head(1).asInstanceOf[Map[String, String]]
    map.get(Gender) shouldBe Some(Female)
  }

  it should "identify the gender of a multiple full name entries (with varying token lengths) correctly" in {
    // noinspection SpellCheckingInspection
    // scalastyle:off
    val (_, _, model, result) = identifyName(Seq(
      "Sherrod Brown",
      "Maria Cantwell",
      "Benjamin L. Cardin",
      "Lisa Maria Blunt Rochester",
      "Thomas Robert Carper",
      "Jennifer González-Colón"
    ).toText)
    // scalastyle:on
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
    val resultingMaps = result.collect().toSeq.map(row => row.get(1)).asInstanceOf[Seq[Map[String, String]]]
    val identifiedGenders = resultingMaps.map(_.get(Gender))
    identifiedGenders shouldBe Seq(Some(Male), Some(Female), Some(Male), Some(Female), Some(Male), Some(Female))
  }

  it should "identify the gender of multiple full name entries by finding honorifics" in {
    // noinspection SpellCheckingInspection
    // scalastyle:off
    val (_, _, model, result) = identifyName(Seq(
      "Mr. Sherrod Brown",
      "Mrs. Maria Cantwell",
      "Mr. Benjamin L. Cardin",
      "Ms. Lisa Maria Blunt Rochester",
      "Mister Thomas Robert Carper",
      "Miss Jennifer González-Colón"
    ).toText)
    // scalastyle:on
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
    model.asInstanceOf[HumanNameDetectorModel[Text]].orderedGenderDetectStrategies.headOption shouldBe
      Some(GenderDetectStrategy.FindHonorific())

    val resultingMaps = result.collect().toSeq.map(row => row.get(1)).asInstanceOf[Seq[Map[String, String]]]
    val identifiedGenders = resultingMaps.map(_.get(Gender))
    identifiedGenders shouldBe Seq(Some(Male), Some(Female), Some(Male), Some(Female), Some(Male), Some(Female))
  }

  it should "not use the honorific strategy to find gender when there are multiple honorifics per entry" in {
    // noinspection SpellCheckingInspection
    // scalastyle:off
    val (_, _, model, _) = identifyName(Seq(
      "Jennifer González-Colón (Miss) (Mr.)"
    ).toText)
    // scalastyle:on
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
    model.asInstanceOf[HumanNameDetectorModel[Text]].orderedGenderDetectStrategies.headOption should not be
      Some(GenderDetectStrategy.FindHonorific())
  }

  it should
    """identify the gender of multiple full name entries in `LastName, FirstName` patterns""".stripMargin in {
    // noinspection SpellCheckingInspection
    // scalastyle:off
    val (_, _, model, result) = identifyName(Seq(
      "Brown, Sherrod",
      "Cantwell, Maria",
      "Cardin, Benjamin",
      "Rochester, Lisa",
      "Carper, Thomas",
      "González-Colón, Jennifer"
    ).toText)
    // scalastyle:on
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true

    val resultingMaps = result.collect().toSeq.map(row => row.get(1)).asInstanceOf[Seq[Map[String, String]]]
    val identifiedGenders = resultingMaps.map(_.get(Gender))
    identifiedGenders shouldBe Seq(Some(Male), Some(Female), Some(Male), Some(Female), Some(Male), Some(Female))
  }

  it should
    """identify the gender of multiple full name entries by using RegEx
      |to detect `LastName, FirstName MiddleNames` patterns""".stripMargin in {
    // noinspection SpellCheckingInspection
    // scalastyle:off
    val (_, _, model, result) = identifyName(Seq(
      "Brown, Sherrod",
      "Cantwell, Maria",
      "Cardin, Benjamin L.",
      "Rochester, Lisa Maria Blunt",
      "Carper, Thomas Robert",
      "González-Colón, Jennifer"
    ).toText)
    // scalastyle:on
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
    model.asInstanceOf[HumanNameDetectorModel[Text]].orderedGenderDetectStrategies.headOption.map(_.entryName) shouldBe
      Some(GenderDetectStrategy.ByRegex(NameDetectUtils.TextAfterFirstComma).entryName)

    val resultingMaps = result.collect().toSeq.map(row => row.get(1)).asInstanceOf[Seq[Map[String, String]]]
    val identifiedGenders = resultingMaps.map(_.get(Gender))
    identifiedGenders shouldBe Seq(Some(Male), Some(Female), Some(Male), Some(Female), Some(Male), Some(Female))
  }

  it should
    """identify the gender of multiple full name entries by using RegEx
      |to detect `LastName, Honorific FirstName MiddleNames` patterns
      |when the honorifics do not convey gender information""".stripMargin in {
    // noinspection SpellCheckingInspection
    // scalastyle:off
    val (_, _, model, result) = identifyName(Seq(
      "Brown, Dr. Sherrod L.",
      "Cantwell, Prof. Maria Blunt"
    ).toText)
    // scalastyle:on
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
    model.asInstanceOf[HumanNameDetectorModel[Text]].orderedGenderDetectStrategies.headOption.map(_.entryName) shouldBe
      Some(GenderDetectStrategy.ByRegex(NameDetectUtils.TextAfterFirstCommaAndNextToken).entryName)

    val resultingMaps = result.collect().toSeq.map(row => row.get(1)).asInstanceOf[Seq[Map[String, String]]]
    val identifiedGenders = resultingMaps.map(_.get(Gender))
    identifiedGenders shouldBe Seq(Some(Male), Some(Female))
  }

  it should "use mixed strategies to detect gender" in {
    // noinspection SpellCheckingInspection
    // scalastyle:off
    val (_, _, model, result) = identifyName(Seq(
      "Sherrod Brown",
      "Cantwell, Maria",
      "Mr. Benjamin L. Cardin",
      "Rochester, Lisa Maria Blunt",
      "Carper, Dr. Thomas Robert",
      "González-Colón, Ms. Jennifer"
    ).toText)
    // scalastyle:on
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
    val resultingMaps = result.collect().toSeq.map(row => row.get(1)).asInstanceOf[Seq[Map[String, String]]]
    val identifiedGenders = resultingMaps.map(_.get(Gender))
    identifiedGenders shouldBe Seq(Some(Male), Some(Female), Some(Male), Some(Female), Some(Male), Some(Female))
  }

  it should "ignore null values in calculating stats" in {
    val names = NameDictionaryGroundTruth.withProbabilityOfEmpty(0.90).take(200).toList
    val (newData, newFeature) = TestFeatureBuilder(names)
    val model = estimator.setInput(newFeature).fit(newData)
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true

    val countNullsModel = estimator.setIgnoreNulls(false).setInput(newFeature).fit(newData)
    countNullsModel.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe false
  }

  it should "produce the correct metadata" in {
    val text = Text("Elizabeth Warren")
    val (_, _, model, _) = identifyName(Seq(text))
    val metadata: Metadata = model.getMetadata()
    metadata shouldBe HumanNameDetectorMetadata(treatAsName = true, predictedNameProb = 1.0,
      genderResultsByStrategy = estimator.computeGenderResultsByStrategy(
        text.value, estimator.preProcess(text), NameDetectUtils.DefaultGenderDictionary)
    ).toMetadata()
  }

  it should "have correctly working metadata helpers" in {
    // noinspection SpellCheckingInspection
    // scalastyle:off
    val (_, _, model, _) = identifyName(Seq(
      "Sherrod Brown",
      "Cantwell, Maria",
      "Mr. Benjamin L. Cardin",
      "Rochester, Lisa Maria Blunt",
      "Carper, Dr. Thomas Robert",
      "González-Colón, Ms. Jennifer"
    ).toText)
    // scalastyle:on
    val metadata: Metadata = model.getMetadata()
    metadata shouldBe HumanNameDetectorMetadata.fromMetadata(metadata).toMetadata()
  }
}
