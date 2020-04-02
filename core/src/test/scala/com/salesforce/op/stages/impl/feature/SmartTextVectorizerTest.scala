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

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.Assertion
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SmartTextVectorizerTest
  extends OpEstimatorSpec[OPVector, SequenceModel[Text, OPVector], SmartTextVectorizer[Text]] with AttributeAsserts {

  lazy val (inputData, f1, f2) = TestFeatureBuilder("text1", "text2",
    Seq[(Text, Text)](
      ("hello world".toText, "Hello world!".toText),
      ("hello world".toText, "What's up".toText),
      ("good evening".toText, "How are you doing, my friend?".toText),
      ("hello world".toText, "Not bad, my friend.".toText),
      (Text.empty, Text.empty)
    )
  )
  val estimator = new SmartTextVectorizer()
    .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1)
    .setTopK(2).setPrependFeatureName(false)
    .setHashSpaceStrategy(HashSpaceStrategy.Shared)
    .setInput(f1, f2)
  val expectedResult = Seq(
    Vectors.sparse(9, Array(0, 4, 6), Array(1.0, 1.0, 1.0)),
    Vectors.sparse(9, Array(0, 8), Array(1.0, 1.0)),
    Vectors.sparse(9, Array(1, 6), Array(1.0, 1.0)),
    Vectors.sparse(9, Array(0, 6), Array(1.0, 2.0)),
    Vectors.sparse(9, Array(3, 8), Array(1.0, 1.0))
  ).map(_.toOPVector)

  /*
    Generate some more complicated input data to check things a little closer. There are four text fields with
    different token distributions:

    country: Uniformly distributed from a larger list of ~few hundred countries, should be hashed
    categoricalText: Uniformly distributed from a small list of choices, should be pivoted (also has fixed lengths,
      so serves as a test that the categorical check happens before the token length variance check)
    textId: Uniformly distributed high cardinality Ids with fixed lengths, should be ignored
    text: Uniformly distributed unicode strings with lengths ranging from 0-100, should be hashed
   */
  val countryData: Seq[Text] = RandomText.countries.withProbabilityOfEmpty(0.2).limit(1000)
  val categoricalTextData: Seq[Text] = RandomText.textFromDomain(domain = List("A", "B", "C", "D", "E", "F"))
    .withProbabilityOfEmpty(0.2).limit(1000)
  // Generate List containing elements like 040231, 040232, ...
  val textIdData: Seq[Text] = RandomText.textFromDomain(
    domain = (1 to 1000).map(x => "%06d".format(40230 + x)).toList
  ).withProbabilityOfEmpty(0.2).limit(1000)
  val textData: Seq[Text] = RandomText.strings(minLen = 0, maxLen = 100).withProbabilityOfEmpty(0.2).limit(1000)
  val generatedData: Seq[(Text, Text, Text, Text)] =
    countryData.zip(categoricalTextData).zip(textIdData).zip(textData).map {
      case (((co, ca), id), te) => (co, ca, id, te)
    }
  val (rawDF, rawCountry, rawCategorical, rawTextId, rawText) = TestFeatureBuilder(
    "country", "categorical", "textId", "text", generatedData)

  val stringData: String = "I have got a LovEly buncH of cOcOnuts. " +
    "Here they are ALL standing in a row."
  val tol = 1e-12 // Tolerance for comparing real numbers

  it should "detect one categorical and one non-categorical text feature" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val categoricalVectorized = new OpTextPivotVectorizer[Text]().setMinSupport(1).setTopK(2).setInput(f1).getOutput()
    val tokenizedText = new TextTokenizer[Text]().setInput(f2).getOutput()
    val textVectorized = new OPCollectionHashingVectorizer[TextList]()
      .setNumFeatures(4).setPrependFeatureName(false).setInput(tokenizedText).getOutput()
    val nullIndicator = new TextListNullTransformer[TextList]().setInput(tokenizedText).getOutput()

    val transformed = new OpWorkflow()
      .setResultFeatures(smartVectorized, categoricalVectorized, textVectorized, nullIndicator).transform(inputData)
    val result = transformed.collect(smartVectorized, categoricalVectorized, textVectorized, nullIndicator)
    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true, transformed.collect(smartVectorized))
    val fieldCategorical = transformed.schema(categoricalVectorized.name)
    val catRes = transformed.collect(categoricalVectorized)
    assertNominal(fieldCategorical, Array.fill(catRes.head.value.size)(true), catRes)
    val fieldText = transformed.schema(textVectorized.name)
    val textRes = transformed.collect(textVectorized)
    assertNominal(fieldText, Array.fill(textRes.head.value.size)(false), textRes)
    val (smart, expected) = result.map { case (smartVector, categoricalVector, textVector, nullVector) =>
      val combined = categoricalVector.combine(textVector, nullVector)
      smartVector -> combined
    }.unzip

    smart shouldBe expected
  }

  it should "detect two categorical text features" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(10).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val categoricalVectorized =
      new OpTextPivotVectorizer[Text]().setMinSupport(1).setTopK(2).setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized, categoricalVectorized).transform(inputData)
    val result = transformed.collect(smartVectorized, categoricalVectorized)
    val field = transformed.schema(smartVectorized.name)
    val smartRes = transformed.collect(smartVectorized)
    assertNominal(field, Array.fill(smartRes.head.value.size)(true), smartRes)
    val fieldCategorical = transformed.schema(categoricalVectorized.name)
    val catRes = transformed.collect(categoricalVectorized)
    assertNominal(fieldCategorical, Array.fill(catRes.head.value.size)(true), catRes)
    val (smart, expected) = result.unzip

    smart shouldBe expected
  }

  it should "detect two non categorical text features" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val f1Tokenized = new TextTokenizer[Text]().setInput(f1).getOutput()
    val f2Tokenized = new TextTokenizer[Text]().setInput(f2).getOutput()
    val textVectorized = new OPCollectionHashingVectorizer[TextList]()
      .setNumFeatures(4).setPrependFeatureName(false).setInput(f1Tokenized, f2Tokenized).getOutput()
    val nullIndicator = new TextListNullTransformer[TextList]().setInput(f1Tokenized, f2Tokenized).getOutput()

    val transformed = new OpWorkflow()
      .setResultFeatures(smartVectorized, textVectorized, nullIndicator).transform(inputData)
    val result = transformed.collect(smartVectorized, textVectorized, nullIndicator)
    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(8)(false) ++ Array(true, true), transformed.collect(smartVectorized))
    val fieldText = transformed.schema(textVectorized.name)
    val textRes = transformed.collect(textVectorized)
    assertNominal(fieldText, Array.fill(textRes.head.value.size)(false), textRes)
    val (smart, expected) = result.map { case (smartVector, textVector, nullVector) =>
      val combined = textVector.combine(nullVector)
      smartVector -> combined
    }.unzip

    smart shouldBe expected
  }

  it should "work for shortcut" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2)
      .setAutoDetectLanguage(false).setMinTokenLength(1).setToLowercase(false)
      .setInput(f1, f2).getOutput()

    val shortcutVectorized = f1.smartVectorize(
      maxCategoricalCardinality = 2, numHashes = 4, minSupport = 1, topK = 2,
      autoDetectLanguage = false, minTokenLength = 1, toLowercase = false,
      hashSpaceStrategy = HashSpaceStrategy.Shared, others = Array(f2)
    )

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized, shortcutVectorized).transform(inputData)
    val result = transformed.collect(smartVectorized, shortcutVectorized)
    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true, transformed.collect(smartVectorized))
    val fieldShortcut = transformed.schema(shortcutVectorized.name)
    assertNominal(fieldShortcut, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true,
      transformed.collect(shortcutVectorized))
    val (regular, shortcut) = result.unzip

    regular shouldBe shortcut
  }

  it should "detect and ignore fields that looks like machine-generated IDs by having a low token length variance" in {
    val topKCategorial = 3
    val hashSize = 5

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(10).setNumFeatures(hashSize).setMinSupport(10).setTopK(topKCategorial)
      .setMinLengthStdDev(0.5).setTextLengthType(TextLengthType.Tokens)
      .setAutoDetectLanguage(false).setMinTokenLength(1).setToLowercase(false)
      .setTrackNulls(true).setTrackTextLen(true)
      .setInput(rawCountry, rawCategorical, rawTextId, rawText).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(rawDF)
    val result = transformed.collect(smartVectorized)

    /*
      Feature vector should have 16 components, corresponding to two hashed text fields, one categorical field, and
      one ignored text field.

      Hashed text: (5 hash buckets + 1 length + 1 null indicator) = 7 elements
      Categorical: (3 topK + 1 other + 1 null indicator) = 5 elements
      Ignored text: (1 length + 1 null indicator) = 2 elements
     */
    val featureVectorSize = 2 * (hashSize + 2) + (topKCategorial + 2) + 2
    val firstRes = result.head
    firstRes.v.size shouldBe featureVectorSize

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.columns.length shouldBe featureVectorSize
    meta.columns.slice(0, 5).forall(_.grouping.contains("categorical"))
    meta.columns.slice(5, 10).forall(_.grouping.contains("country"))
    meta.columns.slice(10, 15).forall(_.grouping.contains("text"))
    meta.columns.slice(15, 18).forall(_.descriptorValue.contains(OpVectorColumnMetadata.TextLenString))
    meta.columns.slice(18, 21).forall(_.indicatorValue.contains(OpVectorColumnMetadata.NullString))
  }

  it should "detect and ignore fields that looks like machine-generated IDs by having a low value length variance" in {
    val topKCategorial = 3
    val hashSize = 5

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(10).setNumFeatures(hashSize).setMinSupport(10).setTopK(topKCategorial)
      .setMinLengthStdDev(1.0).setTextLengthType(TextLengthType.FullEntry)
      .setAutoDetectLanguage(false).setMinTokenLength(1).setToLowercase(false)
      .setTrackNulls(true).setTrackTextLen(true)
      .setInput(rawCountry, rawCategorical, rawTextId, rawText).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(rawDF)
    val result = transformed.collect(smartVectorized)

    /*
      Feature vector should have 16 components, corresponding to two hashed text fields, one categorical field, and
      one ignored text field.

      Hashed text: (5 hash buckets + 1 length + 1 null indicator) = 7 elements
      Categorical: (3 topK + 1 other + 1 null indicator) = 5 elements
      Ignored text: (1 length + 1 null indicator) = 2 elements
     */
    val featureVectorSize = 2 * (hashSize + 2) + (topKCategorial + 2) + 2
    val firstRes = result.head
    firstRes.v.size shouldBe featureVectorSize

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.columns.length shouldBe featureVectorSize
    meta.columns.slice(0, 5).forall(_.grouping.contains("categorical"))
    meta.columns.slice(5, 10).forall(_.grouping.contains("country"))
    meta.columns.slice(10, 15).forall(_.grouping.contains("text"))
    meta.columns.slice(15, 18).forall(_.descriptorValue.contains(OpVectorColumnMetadata.TextLenString))
    meta.columns.slice(18, 21).forall(_.indicatorValue.contains(OpVectorColumnMetadata.NullString))
  }

  it should "fail with an error" in {
    val emptyDF = inputData.filter(inputData("text1") === "").toDF()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val thrown = intercept[IllegalArgumentException] {
      new OpWorkflow().setResultFeatures(smartVectorized).transform(emptyDF)
    }
    assert(thrown.getMessage.contains("requirement failed"))
  }

  it should "generate metadata correctly" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 9
    meta.columns.foreach { col =>
      if (col.index < 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
      } else if (col.index == 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(TransmogrifierDefaults.OtherString)
      } else if (col.index == 3) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe None
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "generate categorical metadata correctly" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(4).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 8
    meta.columns.foreach { col =>
      if (col.index < 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
      } else if (col.index == 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(TransmogrifierDefaults.OtherString)
      } else if (col.index == 3) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else if (col.index < 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
      } else if (col.index == 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(TransmogrifierDefaults.OtherString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "generate non categorical metadata correctly" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 10
    meta.columns.foreach { col =>
      if (col.index < 4) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe None
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe None
      } else if (col.index == 8) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "append the text lengths to the feature vector if one feature is determined to be text" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setTrackTextLen(true).setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)
    val res = transformed.collect(smartVectorized)
    val expected = Array(
      Vectors.sparse(10, Array(0, 4, 6, 8), Array(1.0, 1.0, 1.0, 10.0)),
      Vectors.sparse(10, Array(0, 9), Array(1.0, 1.0)),
      Vectors.sparse(10, Array(1, 6, 8), Array(1.0, 1.0, 6.0)),
      Vectors.sparse(10, Array(0, 6, 8), Array(1.0, 2.0, 9.0)),
      Vectors.sparse(10, Array(3, 9), Array(1.0, 1.0))
    ).map(_.toOPVector)

    res shouldEqual expected

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 10
    meta.columns.foreach { col =>
      if (col.index < 4) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe None
      } else if (col.index == 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe None
        col.descriptorValue shouldBe Option(OpVectorColumnMetadata.TextLenString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "append one text lengths column to the feature vector for each feature determined to be text" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setTrackTextLen(true).setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(inputData)
    val res = transformed.collect(smartVectorized)
    val expected = Array(
      Vectors.sparse(12, Array(0, 2, 4, 6, 8, 9), Array(1.0, 1.0, 1.0, 1.0, 10.0, 10.0)),
      Vectors.sparse(12, Array(0, 2, 8, 11), Array(1.0, 1.0, 10.0, 1.0)),
      Vectors.sparse(12, Array(0, 3, 6, 8, 9), Array(1.0, 1.0, 1.0, 11.0, 6.0)),
      Vectors.sparse(12, Array(0, 2, 6, 8, 9), Array(1.0, 1.0, 2.0, 10.0, 9.0)),
      Vectors.sparse(12, Array(10, 11), Array(1.0, 1.0))
    ).map(_.toOPVector)

    res shouldEqual expected

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 12

    meta.columns(1)

    meta.columns.foreach { col =>
      if (col.index < 4) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe None
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe None
      } else if (col.index == 8) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe None
        col.descriptorValue shouldBe Option(OpVectorColumnMetadata.TextLenString)
      } else if (col.index == 9) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe None
        col.descriptorValue shouldBe Option(OpVectorColumnMetadata.TextLenString)
      } else if (col.index == 10) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "tokenize text correctly using the shortcut used in computeTextStats" in {
    val tokens: TextList = TextTokenizer.tokenizeString(stringData).tokens
    tokens.value should contain theSameElementsAs Seq("got", "lovely", "bunch", "coconuts", "standing", "row")
  }

  it should "turn a string into a corresponding TextStats instance with cleaning" in {
    val res = TextStats.textStatsFromString(stringData, shouldCleanText = true, shouldTokenize = true,
      maxCardinality = 50)
    val tokens: TextList = TextTokenizer.tokenizeString(stringData).tokens

    res.valueCounts.size shouldBe 1
    res.valueCounts should contain ("IHaveGotALovelyBunchOfCoconutsHereTheyAreAllStandingInARow" -> 1)

    res.lengthCounts.size shouldBe tokens.value.map(_.length).distinct.length
    res.lengthCounts should contain (6 -> 1L)
    res.lengthCounts should contain (3 -> 2L)
    res.lengthCounts should contain (5 -> 1L)
    res.lengthCounts should contain (8 -> 2L)

    checkDerivedQuantities(res, Seq(6, 3, 3, 5, 8, 8).map(_.toLong))
  }

  it should "turn a string into a corresponding TextStats instance without cleaning" in {
    val res = TextStats.textStatsFromString(stringData, shouldCleanText = false, shouldTokenize = true,
      maxCardinality = 50)
    val tokens: TextList = TextTokenizer.tokenizeString(stringData).tokens

    res.valueCounts.size shouldBe 1
    res.valueCounts should contain ("I have got a LovEly buncH of cOcOnuts. Here they are ALL standing in a row." -> 1)

    res.lengthCounts.size shouldBe tokens.value.map(_.length).distinct.length
    res.lengthCounts should contain (6 -> 1L)
    res.lengthCounts should contain (3 -> 2L)
    res.lengthCounts should contain (5 -> 1L)
    res.lengthCounts should contain (8 -> 2L)

    checkDerivedQuantities(res, Seq(6, 3, 3, 5, 8, 8).map(_.toLong))
  }

  it should "turn a string into a corresponding TextStats instance that respects maxCardinality" in {
    val tinyCard = 2
    val res = TextStats.textStatsFromString(stringData, shouldCleanText = false, shouldTokenize = true,
      maxCardinality = 2)

    res.valueCounts.size shouldBe 1
    res.valueCounts should contain ("I have got a LovEly buncH of cOcOnuts. Here they are ALL standing in a row." -> 1)

    // MaxCardinality will stop counting as soon as the lengths are > maxCardinality, so the length counts will
    // have maxCardinality + 1 elements, however they will stop being appended to even if future elements have
    // the same key.
    res.lengthCounts.size shouldBe tinyCard + 1
    res.lengthCounts should contain (6 -> 1L)
    res.lengthCounts should contain (3 -> 1L)
    res.lengthCounts should contain (5 -> 1L)

    checkDerivedQuantities(res, Seq(6, 3, 5).map(_.toLong))
  }

  it should "allow toggling of tokenization for calculating the length distribution" in {
    val res = TextStats.textStatsFromString(stringData, shouldCleanText = true, shouldTokenize = false,
      maxCardinality = 50)

    res.valueCounts.size shouldBe 1
    res.valueCounts should contain ("IHaveGotALovelyBunchOfCoconutsHereTheyAreAllStandingInARow" -> 1)
    res.lengthCounts.size shouldBe 1
    res.lengthCounts should contain (58 -> 1L)

    checkDerivedQuantities(res, Seq(58).map(_.toLong))
  }

  /**
   * Set of tests to check that the derived quantities calculated on the length distribution in TextStats
   * match the actual length distributions of the tokens.
   *
   * @param res       TextStats result to compare
   * @param lengthSeq Expected length sequence
   * @return          Assertions on derived quantities in TextStats
   */
  private[op] def checkDerivedQuantities(res: TextStats, lengthSeq: Seq[Long]): Assertion = {
    val expectedLengthMean = lengthSeq.sum.toDouble / lengthSeq.length
    val expectedLengthVariance = lengthSeq.map(x => math.pow((x - expectedLengthMean), 2)).sum / lengthSeq.length
    val expectedLengthStdDev = math.sqrt(expectedLengthVariance)
    res.lengthSize shouldBe lengthSeq.length
    compareWithTol(res.lengthMean, expectedLengthMean, tol)
    compareWithTol(res.lengthVariance, expectedLengthVariance, tol)
    compareWithTol(res.lengthStdDev, expectedLengthStdDev, tol)
  }

  Spec[TextStats] should "aggregate correctly" in {
    val l1 = TextStats(Map("hello" -> 1, "world" -> 2), Map(5 -> 3), TextStats.hllMonoid.zero)
    val r1 = TextStats(Map("hello" -> 1, "world" -> 1), Map(5 -> 2), TextStats.hllMonoid.zero)
    val expected1 = TextStats(Map("hello" -> 2, "world" -> 3), Map(5 -> 5), TextStats.hllMonoid.zero)

    val l2 = TextStats(
      Map("hello" -> 1, "world" -> 2, "ocean" -> 3), Map(5 -> 6), TextStats.hllMonoid.zero
    )
    val r2 = TextStats(Map("hello" -> 1), Map(5 -> 1), TextStats.hllMonoid.zero)
    val expected2 = TextStats(
      Map("hello" -> 1, "world" -> 2, "ocean" -> 3), Map(5 -> 7), TextStats.hllMonoid.zero
    )

    TextStats.monoid(2).plus(l1, r1) shouldBe expected1
    TextStats.monoid(2).plus(l2, r2) shouldBe expected2
  }

  it should "compute correct statistics on the length distributions" in {
    val ts = TextStats(
      Map("hello" -> 2, "joe" -> 2, "woof" -> 1), Map(3 -> 2, 4 -> 1, 5 -> 2), TextStats.hllMonoid.zero
    )

    ts.lengthSize shouldBe 5
    ts.lengthMean shouldBe 4.0
    ts.lengthVariance shouldBe 0.8
    ts.lengthStdDev shouldBe math.sqrt(0.8)
  }

  it should "return sane results when entries do not tokenize and result in empty maps" in {
    val ts = TextStats(Map("the" -> 10, "and" -> 4), Map.empty[Int, Long])

    ts.lengthSize shouldBe 0
    ts.lengthMean.isNaN shouldBe true
    ts.lengthVariance.isNaN shouldBe true
    ts.lengthStdDev.isNaN shouldBe true
  }

}
