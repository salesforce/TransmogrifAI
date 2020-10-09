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
import com.salesforce.op.features.{Feature, FeatureLike}
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.TextVectorizationMethod._
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.stages.{NameDetectUtils, SensitiveFeatureMode}
import org.apache.log4j.Level
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.Assertion
import org.scalatest.junit.JUnitRunner

import scala.util.Random

@RunWith(classOf[JUnitRunner])
class SmartTextMapVectorizerTest
  extends OpEstimatorSpec[OPVector, SequenceModel[TextMap, OPVector], SmartTextMapVectorizer[TextMap]]
    with AttributeAsserts {

  Random.setSeed(42)

  lazy val (inputData, m1, m2, f1, f2) = TestFeatureBuilder("textMap1", "textMap2", "text1", "text2",
    Seq[(TextMap, TextMap, Text, Text)](
      (TextMap(Map("text1" -> "hello world", "text2" -> "Hello world!")), TextMap.empty,
        "hello world".toText, "Hello world!".toText),
      (TextMap(Map("text1" -> "hello world", "text2" -> "What's up")), TextMap.empty,
        "hello world".toText, "What's up".toText),
      (TextMap(Map("text1" -> "good evening", "text2" -> "How are you doing, my friend?")), TextMap.empty,
        "good evening".toText, "How are you doing, my friend?".toText),
      (TextMap(Map("text1" -> "hello world", "text2" -> "Not bad, my friend.")), TextMap.empty,
        "hello world".toText, "Not bad, my friend.".toText),
      (TextMap.empty, TextMap.empty, Text.empty, Text.empty)
    )
  )

  lazy val (data2, m3, m4, f3, f4) = TestFeatureBuilder("textMap1", "textMap2", "text1", "text2",
    Seq[(TextAreaMap, TextAreaMap, Text, Text)](
      (TextAreaMap(Map("text1" -> "hello world", "text2" -> "Hello world!")), TextAreaMap.empty,
        "hello world".toTextArea, "Hello world!".toTextArea),
      (TextAreaMap(Map("text1" -> "hello world", "text2" -> "What's up")), TextAreaMap.empty,
        "hello world".toTextArea, "What's up".toTextArea),
      (TextAreaMap(Map("text1" -> "good evening", "text2" -> "How are you doing, my friend?")), TextAreaMap.empty,
        "good evening".toTextArea, "How are you doing, my friend?".toTextArea),
      (TextAreaMap(Map("text1" -> "hello world", "text2" -> "Not bad, my friend.")), TextAreaMap.empty,
        "hello world".toTextArea, "Not bad, my friend.".toTextArea),
      (TextAreaMap.empty, TextAreaMap.empty, TextArea.empty, TextArea.empty)
    )
  )

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

  def mapifyText(textSeq: Seq[Text]): TextMap = {
    textSeq.zipWithIndex.flatMap {
      case (t, i) => t.value.map(tv => "f" + i.toString -> tv)
    }.toMap.toTextMap
  }
  val mapData = generatedData.map { case (t1, t2, t3, t4) => mapifyText(Seq(t1, t2, t3, t4)) }
  val (rawDF, rawTextMap) = TestFeatureBuilder("textMap", mapData)

  // Do the same thing with the data spread across many maps to test that they get combined correctly as well
  val mapDataSeparate = generatedData.map {
    case (t1, t2, t3, t4) => (mapifyText(Seq(t1, t2)), mapifyText(Seq(t3)), mapifyText(Seq(t4)))
  }
  val (rawDFSeparateMaps, rawTextMap1, rawTextMap2, rawTextMap3) =
    TestFeatureBuilder("textMap1", "textMap2", "textMap3", mapDataSeparate)

  val textMapData = Map(
    "f1" -> "I have got a lovely bunch of coconuts. Here they are all standing in a row.",
    "f2" -> "Olly wolly polly woggy ump bump fizz!"
  )
  val tokensMap = textMapData.mapValues(s => TextTokenizer.tokenizeString(s).tokens)
  val tol = 1e-12 // Tolerance for comparing real numbers

  val oneCountryData: Seq[Text] = Seq.fill(1000)(Text("United States"))
  val categoricalCountryData = Random.shuffle(oneCountryData ++ countryData)
  val countryMapData = categoricalCountryData.map { case country => mapifyText(Seq(country)) }
  val (countryMapDF, rawCatCountryMap) = TestFeatureBuilder("rawCatCountryMap", countryMapData)

  /**
   * Estimator instance to be tested
   */
  override val estimator: SmartTextMapVectorizer[TextMap] = new SmartTextMapVectorizer[TextMap]()
    .setInput(m1, m2)

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  override val expectedResult: Seq[OPVector] = Seq(
    Vectors.dense(Array(1.0, 0.0, 1.0, 0.0)),
    Vectors.dense(Array(1.0, 0.0, 1.0, 0.0)),
    Vectors.dense(Array(1.0, 0.0, 1.0, 0.0)),
    Vectors.dense(Array(1.0, 0.0, 1.0, 0.0)),
    Vectors.dense(Array(0.0, 1.0, 0.0, 1.0))
  ).map(_.toOPVector)

  import spark.sqlContext.implicits._

  Spec[TextMapStats] should "provide a proper semigroup" in {
    val data = Seq(
      TextMapStats(Map(
        "f1" -> TextStats(Map("hello" -> 2, "world" -> 1), Map(5 -> 3)),
        "f2" -> TextStats(Map("hello" -> 2, "ocean" -> 2), Map(5 -> 4)),
        "f3" -> TextStats(Map("foo" -> 1), Map(3 -> 1))
      )),
      TextMapStats(Map(
        "f1" -> TextStats(Map("hello" -> 1), Map(5 -> 1)),
        "f2" -> TextStats(Map("ocean" -> 1, "other" -> 5), Map(5 -> 6))
      )),
      TextMapStats(Map(
        "f2" -> TextStats(Map("other" -> 1), Map(5 -> 1))
      ))
    )
    TextMapStats.monoid(2).sumOption(data) shouldBe Some(TextMapStats(Map(
      "f1" -> TextStats(Map("hello" -> 3, "world" -> 1), Map(5 -> 4)),
      "f2" -> TextStats(Map("hello" -> 2, "ocean" -> 3, "other" -> 5), Map(5 -> 11)),
      "f3" -> TextStats(Map("foo" -> 1), Map(3 -> 1))
    )))
  }

  Spec[SmartTextMapVectorizer[TextMap]] should "detect one categorical and one non-categorical text feature" in {
    val estimator: SmartTextMapVectorizer[TextMap] = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setInput(m1, m2)
    val smartMapVectorized = estimator.getOutput()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, smartVectorized).transform(inputData)
    val result = transformed.collect(smartMapVectorized, smartVectorized)
    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true, transformed.collect(smartVectorized))
    val fieldMap = transformed.schema(smartMapVectorized.name)
    assertNominal(fieldMap, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true,
      transformed.collect(smartMapVectorized))
    val mapMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      if (m.index < 4) m.grouping shouldBe f.grouping
      else m.grouping shouldBe Option(f2.name)
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "detect two categorical text features" in {
    val smartMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(10).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(10).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, smartVectorized).transform(inputData)
    val result = transformed.collect(smartMapVectorized, smartVectorized)
    val field = transformed.schema(smartVectorized.name)
    val rSmart = transformed.collect(smartVectorized)
    assertNominal(field, Array.fill(rSmart.head.value.size)(true), rSmart)
    val fieldMap = transformed.schema(smartMapVectorized.name)
    val rSmartMp = transformed.collect(smartMapVectorized)
    assertNominal(fieldMap, Array.fill(rSmartMp.head.value.size)(true), rSmartMp)
    val mapMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      m.grouping shouldBe f.grouping
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "use separate hash space for each text feature" in {
    val smartMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(1).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setHashSpaceStrategy(HashSpaceStrategy.Separate)
      .setInput(m1, m2).getOutput()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setHashSpaceStrategy(HashSpaceStrategy.Separate)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, smartVectorized).transform(inputData)
    val result = transformed.collect(smartMapVectorized, smartVectorized)
    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(8)(false) ++ Array(true, true), transformed.collect(smartVectorized))
    val fieldMap = transformed.schema(smartMapVectorized.name)
    assertNominal(fieldMap, Array.fill(8)(false) ++ Array(true, true), transformed.collect(smartMapVectorized))
    val mapMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      if (m.index < 4 || m.index == 8) m.grouping shouldBe Option(f1.name)
      else if (m.index < 8 || m.index == 9) m.grouping shouldBe Option(f2.name)
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "use shared hash space for two text features" in {
    val smartMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setNumFeatures(4).setHashSpaceStrategy(HashSpaceStrategy.Shared)
      .setInput(m1, m2).getOutput()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setNumFeatures(4).setHashSpaceStrategy(HashSpaceStrategy.Shared)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, smartVectorized).transform(inputData)
    val result = transformed.collect(smartMapVectorized, smartVectorized)
    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(4)(false) ++ Array(true, true), transformed.collect(smartVectorized))
    val fieldMap = transformed.schema(smartMapVectorized.name)
    assertNominal(fieldMap, Array.fill(4)(false) ++ Array(true, true), transformed.collect(smartMapVectorized))
    val mapMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      if (m.index == 4) {
        assert(m.grouping === Option(f1.name), s"first null indicator should be from ${f1.name}")
      } else if (m.index == 5) {
        assert(m.grouping === Option(f2.name), s"second null indicator should be from ${f2.name}")
      }
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "use shared hash space for two text features again" in {
    val smartMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setNumFeatures(TransmogrifierDefaults.MaxNumOfFeatures).setHashSpaceStrategy(HashSpaceStrategy.Auto)
      .setInput(m1, m2).getOutput()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setNumFeatures(TransmogrifierDefaults.MaxNumOfFeatures).setHashSpaceStrategy(HashSpaceStrategy.Auto)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, smartVectorized).transform(inputData)
    val result = transformed.collect(smartMapVectorized, smartVectorized)
    val field = transformed.schema(smartVectorized.name)
    val rSmart = transformed.collect(smartVectorized)
    assertNominal(field, Array.fill(rSmart.head.value.size - 2)(false) ++ Array(true, true), rSmart)
    val fieldMap = transformed.schema(smartMapVectorized.name)
    val rSmartMp = transformed.collect(smartMapVectorized)
    assertNominal(fieldMap, Array.fill(rSmartMp.head.value.size - 2)(false) ++ Array(true, true), rSmartMp)
    val mapMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      if (m.index == TransmogrifierDefaults.MaxNumOfFeatures) {
        assert(m.grouping === Option(f1.name), s"first null indicator should be from ${f1.name}")
      } else if (m.index > TransmogrifierDefaults.MaxNumOfFeatures) {
        assert(m.grouping === Option(f2.name), s"second null indicator should be from ${f2.name}")
      }
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "product the same result for shortcut" in {
    val smartMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val shortcutMapVectorized = m1.smartVectorize(
      maxCategoricalCardinality = 2, numHashes = 4,
      autoDetectLanguage = false, minTokenLength = 1, toLowercase = false,
      minSupport = 1, topK = 2, prependFeatureName = true,
      others = Array(m2)
    )

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, shortcutMapVectorized).transform(inputData)
    val result = transformed.collect(smartMapVectorized, shortcutMapVectorized)
    val field = transformed.schema(shortcutMapVectorized.name)
    assertNominal(field, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true,
      transformed.collect(shortcutMapVectorized))
    val fieldMap = transformed.schema(smartMapVectorized.name)
    assertNominal(fieldMap, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true,
      transformed.collect(smartMapVectorized))
    val smartMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val shortcutMeta = OpVectorMetadata(transformed.schema(shortcutMapVectorized.name))
    smartMeta.history.keys shouldBe shortcutMeta.history.keys
    smartMeta.columns.length shouldBe shortcutMeta.columns.length

    smartMeta.columns.zip(shortcutMeta.columns).foreach { case (smart, shortcut) =>
      smart.parentFeatureName shouldBe shortcut.parentFeatureName
      smart.parentFeatureType shouldBe shortcut.parentFeatureType
      smart.grouping shouldBe shortcut.grouping
      smart.indicatorValue shouldBe shortcut.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "work on textarea map fields" in {
    val textMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val textAreaMapVectorized = new SmartTextMapVectorizer[TextAreaMap]()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setInput(m3, m4).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(textMapVectorized, textAreaMapVectorized).transform(inputData)
    val result = transformed.collect(textMapVectorized, textAreaMapVectorized)
    val field = transformed.schema(textMapVectorized.name)
    assertNominal(field, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true,
      transformed.collect(textMapVectorized))
    val fieldMap = transformed.schema(textAreaMapVectorized.name)
    assertNominal(fieldMap, Array.fill(4)(true) ++ Array.fill(4)(false) :+ true,
      transformed.collect(textAreaMapVectorized))
    val textMapMeta = OpVectorMetadata(transformed.schema(textMapVectorized.name))
    val textareaMapMeta = OpVectorMetadata(transformed.schema(textAreaMapVectorized.name))
    textMapMeta.history.keys shouldBe textareaMapMeta.history.keys
    textMapMeta.columns.length shouldBe textareaMapMeta.columns.length

    textMapMeta.columns.zip(textareaMapMeta.columns).foreach { case (textMap, textareaMap) =>
      textMap.parentFeatureName shouldBe textareaMap.parentFeatureName
      textMap.parentFeatureType shouldBe Array("com.salesforce.op.features.types.TextMap")
      textareaMap.parentFeatureType shouldBe Array("com.salesforce.op.features.types.TextAreaMap")
      textMap.grouping shouldBe textareaMap.grouping
      textMap.indicatorValue shouldBe textareaMap.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "append lengths of the true text features to the feature vector, if requested" in {
    val smartMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setTrackTextLen(true)
      .setInput(m1, m2).getOutput()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setTrackTextLen(true)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, smartVectorized).transform(inputData)
    val result = transformed.collect(smartMapVectorized, smartVectorized)

    val field = transformed.schema(smartVectorized.name)
    assertNominal(field, Array.fill(4)(true) ++ Array.fill(4)(false) :+ false :+ true,
      transformed.collect(smartVectorized))
    val fieldMap = transformed.schema(smartMapVectorized.name)
    assertNominal(fieldMap, Array.fill(4)(true) ++ Array.fill(4)(false) :+ false :+ true,
      transformed.collect(smartMapVectorized))
    val mapMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      if (m.index < 4) m.grouping shouldBe f.grouping
      else m.grouping shouldBe Option(f2.name)
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "detect and ignore fields that looks like machine-generated IDs by having a low token length variance " +
    "when data is in a single TextMap" in {
    val topKCategorial = 3
    val hashSize = 5

    val smartVectorized = new SmartTextMapVectorizer()
      .setMaxCardinality(10).setNumFeatures(hashSize).setMinSupport(10).setTopK(topKCategorial)
      .setMinLengthStdDev(0.5).setTextLengthType(TextLengthType.Tokens)
      .setAutoDetectLanguage(false).setMinTokenLength(1).setToLowercase(false)
      .setTrackNulls(true).setTrackTextLen(true)
      .setInput(rawTextMap).getOutput()

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

  it should "detect and ignore fields that looks like machine-generated IDs by having a low value length variance " +
    "when data is in a single TextMap" in {
    val topKCategorial = 3
    val hashSize = 5

    val smartVectorized = new SmartTextMapVectorizer()
      .setMaxCardinality(10).setNumFeatures(hashSize).setMinSupport(10).setTopK(topKCategorial)
      .setMinLengthStdDev(1.0).setTextLengthType(TextLengthType.FullEntry)
      .setAutoDetectLanguage(false).setMinTokenLength(1).setToLowercase(false)
      .setTrackNulls(true).setTrackTextLen(true)
      .setInput(rawTextMap).getOutput()

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

  it should "detect and ignore fields that looks like machine-generated IDs by having a low token length variance " +
    "when data is in many TextMaps" in {
    val topKCategorial = 3
    val hashSize = 5

    val smartVectorized = new SmartTextMapVectorizer()
      .setMaxCardinality(10).setNumFeatures(hashSize).setMinSupport(10).setTopK(topKCategorial)
      .setMinLengthStdDev(0.5).setTextLengthType(TextLengthType.Tokens)
      .setAutoDetectLanguage(false).setMinTokenLength(1).setToLowercase(false)
      .setTrackNulls(true).setTrackTextLen(true)
      .setInput(rawTextMap1, rawTextMap2, rawTextMap3).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(rawDFSeparateMaps)
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

  lazy val (newInputData, features) = {
    val N = 5

    val baseText1 = Seq("hello world", "hello world", "good evening").toText ++ Seq(Text.empty, Text.empty)
    val baseText2 = Seq(
      "Hello world!", "What's up", "How are you doing, my friend?", "Not bad, my friend").toText :+ Text.empty
    val baseNames = Seq("Michael", "Michelle", "Roxanne", "Ross").toText :+ Text.empty

    def convertToMap(texts: Seq[Text]): Map[String, String] = texts.map(_.value).zipWithIndex.collect {
      case (Some(text), index) => (if (index == 2) "name" else s"text${index + 1}") -> text } toMap
    val textMap1: Seq[TextMap] = (baseText1, baseText2, baseNames).zipped.map {
      case (a, b, c) => TextMap(convertToMap(Seq(a, b, c)))
    }
    val textMap2: Seq[TextMap] = Seq.fill[TextMap](N)(TextMap.empty)
    val textMap3: Seq[TextMap] = (baseText1, baseText2).zipped.map {
      case (a, b) => TextMap(convertToMap(Seq(a, b)))
    }

    val textAreaMap1: Seq[TextAreaMap] = (baseText1, baseText2, baseNames).zipped.map {
      case (a, b, c) => TextAreaMap(convertToMap(Seq(a, b, c)))
    }
    val textAreaMap2: Seq[TextAreaMap] = Seq.fill[TextAreaMap](N)(TextAreaMap.empty)
    val textAreaMap3: Seq[TextAreaMap] = (baseText1, baseText2).zipped.map {
      case (a, b) => TextAreaMap(convertToMap(Seq(a, b)))
    }

    val nameTextMap: Seq[TextMap] = baseNames.map(_.value match {
      case Some(text) => TextMap(Map("name" -> text))
      case None => TextMap(Map.empty)
    })
    val nameTextAreaMap: Seq[TextAreaMap] = baseNames.map(_.value match {
      case Some(text) => TextAreaMap(Map("name" -> text))
      case None => TextAreaMap(Map.empty)
    })

    val allFeatures = Seq(
      baseText1,       // f0
      baseText2,       // f1
      baseNames,       // f2
      textMap1,        // f3
      textMap2,        // f4
      textAreaMap1,    // f5
      textAreaMap2,    // f6
      nameTextMap,     // f7
      nameTextAreaMap, // f8
      textMap3,        // f9
      textAreaMap3     // f10
    )
    TestFeatureBuilder(allFeatures: _*)
  }

  val newF0: Feature[Text] = features(0).asInstanceOf[Feature[Text]]
  val newF1: Feature[Text] = features(1).asInstanceOf[Feature[Text]]
  val newF2: Feature[Text] = features(2).asInstanceOf[Feature[Text]]
  val newF3: Feature[TextMap] = features(3).asInstanceOf[Feature[TextMap]]
  val newF4: Feature[TextMap] = features(4).asInstanceOf[Feature[TextMap]]
  val newF5: Feature[TextAreaMap] = features(5).asInstanceOf[Feature[TextAreaMap]]
  val newF6: Feature[TextAreaMap] = features(6).asInstanceOf[Feature[TextAreaMap]]
  val newF7: Feature[TextMap] = features(7).asInstanceOf[Feature[TextMap]]
  val newF8: Feature[TextAreaMap] = features(8).asInstanceOf[Feature[TextAreaMap]]
  val newF9: Feature[TextMap] = features(9).asInstanceOf[Feature[TextMap]]
  val newF10: Feature[TextAreaMap] = features(10).asInstanceOf[Feature[TextAreaMap]]

  val biasEstimator: SmartTextVectorizer[Text] = new SmartTextVectorizer()
    .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1)
    .setTopK(2).setPrependFeatureName(false)
    .setHashSpaceStrategy(HashSpaceStrategy.Shared)
    .setSensitiveFeatureMode(SensitiveFeatureMode.DetectAndRemove)
    .setInput(newF0, newF1)

  val biasMapEstimator: SmartTextMapVectorizer[TextMap] = new SmartTextMapVectorizer()
    .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1)
    .setTopK(2).setPrependFeatureName(false)
    .setHashSpaceStrategy(HashSpaceStrategy.Shared)
    .setSensitiveFeatureMode(SensitiveFeatureMode.DetectAndRemove)
    .setInput(newF3, newF4)

  val biasAreaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = new SmartTextMapVectorizer()
    .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1)
    .setTopK(2).setPrependFeatureName(false)
    .setHashSpaceStrategy(HashSpaceStrategy.Shared)
    .setSensitiveFeatureMode(SensitiveFeatureMode.DetectAndRemove)
    .setInput(newF5, newF6)

  Spec[SmartTextMapVectorizer[OPMap[String]]] should "detect a single name feature" in {
    val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF7)
    val mapModel: SmartTextMapVectorizerModel[TextMap] = mapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextMap]]
    mapModel.args.allFeatureInfo.flatMap(_.map(_.vectorizationMethod)) shouldBe Seq(Ignore)

    val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF8)
    val areaMapModel: SmartTextMapVectorizerModel[TextAreaMap] = areaMapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextAreaMap]]
    areaMapModel.args.allFeatureInfo.flatMap(_.map(_.vectorizationMethod)) shouldBe Seq(Ignore)
  }

  it should "detect a single name feature and return empty vectors" in {
    val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF7)
    val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF8)

    val smartVectorized = mapEstimator.getOutput()
    val smartAreaVectorized = areaMapEstimator.getOutput()
    val transformed = new OpWorkflow()
      .setResultFeatures(smartVectorized, smartAreaVectorized).transform(newInputData)
    val result1 = transformed.collect(smartVectorized)
    val result2 = transformed.collect(smartAreaVectorized)

    // The only entries in the output should be null indicators
    val expected = Seq(
      Vectors.dense(Array(0.0)),
      Vectors.dense(Array(0.0)),
      Vectors.dense(Array(0.0)),
      Vectors.dense(Array(0.0)),
      Vectors.dense(Array(1.0))
    ).map(_.toOPVector)

    result1 shouldBe expected
    result2 shouldBe expected

    OpVectorMetadata("OutputVector", mapEstimator.getMetadata()).size shouldBe 1
    OpVectorMetadata("OutputVector", areaMapEstimator.getMetadata()).size shouldBe 1
  }

  it should "detect a single name column among other non-name Text columns" in {
    val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF3, newF4)
    val mapModel: SmartTextMapVectorizerModel[TextMap] = mapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextMap]]
    mapModel.args.allFeatureInfo.flatMap(_.map(_.vectorizationMethod)) shouldBe
      Array(Pivot, Hash, Ignore)

    val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF5, newF6)
    val areaMapModel: SmartTextMapVectorizerModel[TextAreaMap] = areaMapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextAreaMap]]
    areaMapModel.args.allFeatureInfo.flatMap(_.map(_.vectorizationMethod)) shouldBe
      Array(Pivot, Hash, Ignore)
  }

  it should "not create information in the vector for a single name column among other non-name Text columns" in {
    {
      val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF3, newF4)
      val mapOutput = mapEstimator.getOutput()

      val withoutNamesEstimator: SmartTextMapVectorizer[TextMap] = new SmartTextMapVectorizer[TextMap]()
        .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1)
        .setTopK(2).setPrependFeatureName(false)
        .setHashSpaceStrategy(HashSpaceStrategy.Shared)
        .setSensitiveFeatureMode(SensitiveFeatureMode.Off)
        .setInput(newF9, newF4)
      val withoutNamesOutput = withoutNamesEstimator.getOutput()

      val transformed = new OpWorkflow().setResultFeatures(mapOutput, withoutNamesOutput).transform(newInputData)
      val result = transformed.collect(mapOutput, withoutNamesOutput)
      val (withNames, withoutNames) = result.unzip

      // There should only be a single extra metadata entry for the null indicator of the (removed) name field
      OpVectorMetadata("OutputVector", mapEstimator.getMetadata()).size shouldBe
        OpVectorMetadata("OutputVector", withoutNamesEstimator.getMetadata()).size + 1

      // All of the entries in the vector
      // (other than the last one corresponding to the null indicator for the removed name field)
      // should be the same
      // Note: Changing the order of vector information will fail this test
      withNames.zip(withoutNames) foreach { case (withVector, withoutVector) =>
        val withArray = withVector.value.toArray
        val withoutArray = withoutVector.value.toArray :+ withArray.last
        withArray shouldBe withoutArray
      }
    }
    {
      val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF5, newF6)
      val areaMapOutput = areaMapEstimator.getOutput()

      val oldMapEstimator: SmartTextMapVectorizer[TextAreaMap] = new SmartTextMapVectorizer[TextAreaMap]()
        .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1)
        .setTopK(2).setPrependFeatureName(false)
        .setHashSpaceStrategy(HashSpaceStrategy.Shared)
        .setSensitiveFeatureMode(SensitiveFeatureMode.Off)
        .setInput(newF10, newF6)
      val oldMapOutput = oldMapEstimator.getOutput()

      val transformed = new OpWorkflow().setResultFeatures(areaMapOutput, oldMapOutput).transform(newInputData)
      val result = transformed.collect(areaMapOutput, oldMapOutput)
      val (withNames, withoutNames) = result.unzip

      OpVectorMetadata("OutputVector", areaMapEstimator.getMetadata()).size shouldBe
        OpVectorMetadata("OutputVector", oldMapEstimator.getMetadata()).size + 1

      withNames.zip(withoutNames) foreach { case (withVector, withoutVector) =>
        val withArray = withVector.value.toArray
        val withoutArray = withoutVector.value.toArray :+ withArray.last
        withArray shouldBe withoutArray
      }
    }
  }

  it should "compute sensitive information in the metadata for one detected name column" in {
    val prevLoggingLevels = getLoggingLevels
    loggingLevel(Level.DEBUG) // Changes SensitiveFeatureInformation creation logic

    def assertSensitive(estimator: SequenceEstimator[_, _], fname: String): Unit = {
      val sensitive = OpVectorMetadata("OutputVector", estimator.getMetadata()).sensitive
      sensitive.get(fname) match {
        case Some(Seq(SensitiveNameInformation(
          probName, genderDetectResults, probMale, probFemale, probOther, name, mapKey, actionTaken
        ))) =>
          probName shouldBe 1.0
          genderDetectResults.length shouldBe NameDetectUtils.GenderDetectStrategies.length
          probMale shouldBe 0.5
          probFemale shouldBe 0.5
          probOther shouldBe 0.0
          name shouldBe fname
          mapKey shouldBe Some("name")
          actionTaken shouldBe true
        case None => fail("Sensitive information not found in the metadata.")
        case _ => fail("Wrong kind of sensitive information found in the metadata.")
      }
    }

    val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF7)
    val mapModel: SmartTextMapVectorizerModel[TextMap] = mapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextMap]]
    assertSensitive(mapEstimator, newF7.name)

    val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF8)
    val areaMapModel: SmartTextMapVectorizerModel[TextAreaMap] = areaMapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextAreaMap]]
    assertSensitive(areaMapEstimator, newF8.name)

    loggingLevels(prevLoggingLevels) // Reset logging levels
  }

  it should "compute sensitive information in the metadata for multiple detected name columns" in {
    def assertSensitive(estimator: SequenceEstimator[_, _], fname: String, fMapKey: String): Unit = {
      val sensitive = OpVectorMetadata("OutputVector", estimator.getMetadata()).sensitive
      sensitive.get(fname) match {
        case Some(Seq(SensitiveNameInformation(
          probName, genderDetectResults, probMale, probFemale, probOther, name, mapKey, actionTaken
        ))) =>
          probName should be > 0.0
          probName should be <= 1.0
          probMale should be <= 1.0
          probFemale should be <= 1.0
          probOther should be <= 1.0
          (probMale + probFemale + probOther) - 1.0 should be < 0.01

          genderDetectResults.length shouldBe 1 // because debugging is off
          name shouldBe fname
          mapKey shouldBe Some(fMapKey)
          actionTaken shouldBe true
        case None => fail("Sensitive information not found in the metadata.")
        case _ => fail("Wrong kind of sensitive information found in the metadata.")
      }
    }

    val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF3, newF4, newF7)
    val mapModel: SmartTextMapVectorizerModel[TextMap] = mapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextMap]]
    assertSensitive(mapEstimator, newF3.name, "name")
    assertSensitive(mapEstimator, newF7.name, "name")

    val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF5, newF6, newF8)
    val areaMapModel: SmartTextMapVectorizerModel[TextAreaMap] = areaMapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextAreaMap]]
    assertSensitive(areaMapEstimator, newF5.name, "name")
    assertSensitive(areaMapEstimator, newF8.name, "name")
  }

  it should "compute sensitive information in the metadata for all columns only when debugging is on" in {
    val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF3, newF4, newF7)
    val mapModel: SmartTextMapVectorizerModel[TextMap] = mapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextMap]]
    val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF5, newF6, newF8)
    val areaMapModel: SmartTextMapVectorizerModel[TextAreaMap] = areaMapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextAreaMap]]

    val mapSensitive = OpVectorMetadata("OutputVector", mapEstimator.getMetadata()).sensitive
    val areaMapSensitive = OpVectorMetadata("OutputVector", areaMapEstimator.getMetadata()).sensitive

    val sensitiveCols = Seq(newF3, newF7, newF5, newF8).map(_.name)
    Seq(newF3, newF4, newF7).map(_.name) foreach { fname: String =>
      if (sensitiveCols contains fname) {
        mapSensitive contains fname shouldBe true
      } else {
        mapSensitive contains fname shouldBe false
      }
    }
    Seq(newF5, newF6, newF8).map(_.name) foreach { fname: String =>
      if (sensitiveCols contains fname) {
        areaMapSensitive contains fname shouldBe true
      } else {
        areaMapSensitive contains fname shouldBe false
      }
    }

    {
      val prevLoggingLevels = getLoggingLevels
      loggingLevel(Level.DEBUG) // Changes SensitiveFeatureInformation creation logic
      val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF3, newF4, newF7)
      val mapModel: SmartTextMapVectorizerModel[TextMap] = mapEstimator
        .fit(newInputData)
        .asInstanceOf[SmartTextMapVectorizerModel[TextMap]]
      val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF5, newF6, newF8)
      val areaMapModel: SmartTextMapVectorizerModel[TextAreaMap] = areaMapEstimator
        .fit(newInputData)
        .asInstanceOf[SmartTextMapVectorizerModel[TextAreaMap]]

      val mapSensitive = OpVectorMetadata("OutputVector", mapEstimator.getMetadata()).sensitive
      val areaMapSensitive = OpVectorMetadata("OutputVector", areaMapEstimator.getMetadata()).sensitive
      Seq(newF3, newF7).map(_.name) foreach { fname: String =>
        mapSensitive contains fname shouldBe true
        mapSensitive get fname match {
          case Some(Seq(SensitiveNameInformation(_, _, _, _, _, _, _, actionTaken))) =>
            if (sensitiveCols contains fname) {
              actionTaken shouldBe true
            } else {
              actionTaken shouldBe false
            }
          case None => fail("Sensitive information not found in the metadata.")
          case Some(_) => fail("Wrong kind of sensitive information found in the metadata.")
        }
      }
      Seq(newF5, newF8).map(_.name) foreach { fname: String =>
        areaMapSensitive contains fname shouldBe true
        areaMapSensitive get fname match {
          case Some(Seq(SensitiveNameInformation(_, _, _, _, _, _, _, actionTaken))) =>
            if (sensitiveCols contains fname) {
              actionTaken shouldBe true
            } else {
              actionTaken shouldBe false
            }
          case None => fail("Sensitive information not found in the metadata.")
          case Some(_) => fail("Wrong kind of sensitive information found in the metadata.")
        }
      }
      loggingLevels(prevLoggingLevels) // Reset logging levels
    }
  }

  it should "detect and ignore fields that looks like machine-generated IDs by having a low value length variance " +
    "when data is in many TextMaps" in {
    val topKCategorial = 3
    val hashSize = 5

    val smartVectorized = new SmartTextMapVectorizer()
      .setMaxCardinality(10).setNumFeatures(hashSize).setMinSupport(10).setTopK(topKCategorial)
      .setMinLengthStdDev(1.0).setTextLengthType(TextLengthType.FullEntry)
      .setAutoDetectLanguage(false).setMinTokenLength(1).setToLowercase(false)
      .setTrackNulls(true).setTrackTextLen(true)
      .setInput(rawTextMap1, rawTextMap2, rawTextMap3).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(rawDFSeparateMaps)
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

  it should "treat the edge case of coverage being near 0" in {
    val maxCard = 100
    val vectorizer = new SmartTextMapVectorizer().setCoveragePct(1e-10).setMaxCardinality(maxCard).setMinSupport(1)
      .setTrackTextLen(true).setInput(rawCatCountryMap)
    val output = vectorizer.getOutput()
    val transformed = new OpWorkflow().setResultFeatures(output).transform(countryMapDF)
    assertVectorLength(transformed, output, TransmogrifierDefaults.TopK + 2, Pivot)
  }
  it should "treat the edge case of coverage being near 1" in {
    val maxCard = 100
    val vectorizer = new SmartTextMapVectorizer().setCoveragePct(1.0 - 1e-10).setMaxCardinality(maxCard)
      .setMinSupport(1)
      .setTrackTextLen(true).setInput(rawCatCountryMap)
    val output = vectorizer.getOutput()
    val transformed = new OpWorkflow().setResultFeatures(output).transform(countryMapDF)
    assertVectorLength(transformed, output, TransmogrifierDefaults.DefaultNumOfFeatures + 2, Hash)
  }

  it should "detect one categorical with high cardinality using the coverage" in {
    val maxCard = 100
    val topK = 10
    val cardinality = countryMapDF.select(rawCatCountryMap).as[TextMap#Value].map(_("f0")).distinct().count().toInt
    cardinality should be > maxCard
    cardinality should be > topK
    val vectorizer = new SmartTextMapVectorizer()
      .setMaxCardinality(maxCard).setTopK(topK).setMinSupport(1).setCoveragePct(0.5).setCleanText(false)
      .setInput(rawCatCountryMap)
    val output = vectorizer.getOutput()
    val transformed = new OpWorkflow().setResultFeatures(output).transform(countryMapDF)
    assertVectorLength(transformed, output, topK + 2, Pivot)
  }

  it should "not pivot using the coverage because of a high minimum support" in {
    val maxCard = 100
    val topK = 10
    val minSupport = 99999
    val numHashes = 5
    val cardinality = countryMapDF.select(rawCatCountryMap).as[TextMap#Value].map(_("f0")).distinct().count().toInt
    cardinality should be > maxCard
    cardinality should be > topK
    val vectorizer = new SmartTextMapVectorizer()
      .setMaxCardinality(maxCard).setTopK(topK).setMinSupport(minSupport).setNumFeatures(numHashes).setCoveragePct(0.5)
      .setTrackTextLen(true).setCleanText(false).setInput(rawCatCountryMap)
    val output = vectorizer.getOutput()
    val transformed = new OpWorkflow().setResultFeatures(output).transform(countryMapDF)
    assertVectorLength(transformed, output, numHashes + 2, Hash)
  }

  it should "still pivot using the coverage despite a high minimum support" in {
    val maxCard = 100
    val topK = 10
    val minSupport = 100
    val numHashes = 5
    val cardinality = countryMapDF.select(rawCatCountryMap).as[TextMap#Value].map(_("f0")).distinct().count().toInt
    cardinality should be > maxCard
    cardinality should be > topK
    val vectorizer = new SmartTextMapVectorizer()
      .setMaxCardinality(maxCard).setTopK(topK).setMinSupport(minSupport).setNumFeatures(numHashes).setCoveragePct(0.5)
      .setTrackTextLen(true).setCleanText(false).setInput(rawCatCountryMap)
    val output = vectorizer.getOutput()
    val transformed = new OpWorkflow().setResultFeatures(output).transform(countryMapDF)
    assertVectorLength(transformed, output, 3, Pivot)
  }

  it should "not pivot using the coverage because top K is too high" in {
    val maxCard = 100
    val topK = 1000000
    val numHashes = 5
    val cardinality = countryMapDF.select(rawCatCountryMap).as[TextMap#Value].map(_("f0")).distinct().count().toInt
    cardinality should be > maxCard
    cardinality should be <= topK
    val vectorizer = new SmartTextMapVectorizer()
      .setMaxCardinality(maxCard).setTopK(topK).setNumFeatures(numHashes).setCoveragePct(0.5)
      .setTrackTextLen(true).setCleanText(false).setInput(rawCatCountryMap)
    val output = vectorizer.getOutput()
    val transformed = new OpWorkflow().setResultFeatures(output).transform(countryMapDF)
    assertVectorLength(transformed, output, numHashes + 2, Hash)
  }

  it should "still transform country into text, despite the coverage" in {
    val maxCard = 100
    val topK = 10
    val numHashes = 5
    val cardinality = rawDFSeparateMaps.select(rawTextMap1).as[TextMap#Value].map(_.get("f0")).distinct().count().toInt
    cardinality should be > maxCard
    cardinality should be > topK
    val coverageHashed = new SmartTextMapVectorizer()
      .setMaxCardinality(maxCard).setTopK(topK).setMinSupport(1).setCoveragePct(0.5).setCleanText(false)
      .setTrackTextLen(true).setNumFeatures(numHashes).setInput(rawTextMap1).getOutput()
    val transformed = new OpWorkflow().setResultFeatures(coverageHashed).transform(rawDFSeparateMaps)
    val expectedLength = numHashes + 2 + categoricalTextData.toSet.filter(!_.isEmpty).toSeq.length + 2
    assertVectorLength(transformed, coverageHashed, expectedLength , Hash)
  }

  it should "create a TextStats object from text that makes sense" in {
    val res = TextMapStats.computeTextMapStats[TextMap](
      textMapData,
      shouldCleanKeys = false,
      shouldCleanValues = false,
      shouldTokenize = true,
      maxCardinality = 100
    )

    // Check sizes
    res.keyValueCounts.size shouldBe 2 // Two keys in textMapData
    res.keyValueCounts.keySet should contain theSameElementsAs Seq("f1", "f2")

    // Check value counts
    res.keyValueCounts("f1").valueCounts.size shouldBe 1
    res.keyValueCounts("f1").valueCounts should contain
    ("I have got a lovely bunch of coconuts. Here they are all standing in a row." -> 1)
    res.keyValueCounts("f2").valueCounts.size shouldBe 1
    res.keyValueCounts("f2").valueCounts should contain ("Olly wolly polly woggy ump bump fizz!" -> 1)

    // Check token length counts
    res.keyValueCounts("f1").lengthCounts.size shouldBe tokensMap("f1").value.map(_.length).distinct.length
    res.keyValueCounts("f1").lengthCounts should contain (6 -> 1L)
    res.keyValueCounts("f1").lengthCounts should contain (3 -> 2L)
    res.keyValueCounts("f1").lengthCounts should contain (5 -> 1L)
    res.keyValueCounts("f1").lengthCounts should contain (8 -> 2L)
    res.keyValueCounts("f2").lengthCounts.size shouldBe tokensMap("f2").value.map(_.length).distinct.length
    res.keyValueCounts("f2").lengthCounts should contain (4 -> 3L)
    res.keyValueCounts("f2").lengthCounts should contain (5 -> 3L)
    res.keyValueCounts("f2").lengthCounts should contain (3 -> 1L)
  }

  it should "create a TextStats with the correct derived quantities" in {
    val res = TextMapStats.computeTextMapStats[TextMap](
      textMapData,
      shouldCleanKeys = false,
      shouldCleanValues = false,
      shouldTokenize = true,
      maxCardinality = 100
    )

    checkDerivedQuantities(res, "f1", Seq(6, 3, 3, 5, 8, 8).map(_.toLong))
    checkDerivedQuantities(res, "f2", Seq(4, 5, 5, 5, 3, 4, 4).map(_.toLong))
  }

  it should "turn a string into a corresponding TextStats instance that respects maxCardinality" in {
    val res = TextMapStats.computeTextMapStats[TextMap](
      textMapData,
      shouldCleanKeys = false,
      shouldCleanValues = false,
      shouldTokenize = true,
      maxCardinality = 2
    )

    // Check lengths
    res.keyValueCounts.size shouldBe 2 // Two keys in textMapData
    res.keyValueCounts.keySet should contain theSameElementsAs Seq("f1", "f2")

    // Check value counts
    res.keyValueCounts("f1").valueCounts.size shouldBe 1
    res.keyValueCounts("f1").valueCounts should contain
    ("I have got a lovely bunch of coconuts. Here they are all standing in a row." -> 1)
    res.keyValueCounts("f2").valueCounts.size shouldBe 1
    res.keyValueCounts("f2").valueCounts should contain ("Olly wolly polly woggy ump bump fizz!" -> 1)

    // Check token length counts
    res.keyValueCounts("f1").lengthCounts.size shouldBe 3
    res.keyValueCounts("f1").lengthCounts should contain (6 -> 1L)
    res.keyValueCounts("f1").lengthCounts should contain (3 -> 1L)
    res.keyValueCounts("f1").lengthCounts should contain (5 -> 1L)
    checkDerivedQuantities(res, "f1", Seq(6, 3, 5).map(_.toLong))

    res.keyValueCounts("f2").lengthCounts.size shouldBe 3
    res.keyValueCounts("f2").lengthCounts should contain (4 -> 1L)
    res.keyValueCounts("f2").lengthCounts should contain (5 -> 3L)
    res.keyValueCounts("f2").lengthCounts should contain (3 -> 1L)
    checkDerivedQuantities(res, "f2", Seq(4, 5, 5, 5, 3).map(_.toLong))
  }

  it should "turn on stripHTML flag is equivalent to passing in a custom AnalyzerHtmlStrip" +
    "inside SmartTextMapVectorizer" in {
    val exampleHTML = "<body>Big ones, small <h1>ones</h1>, some as big as your head</body>".toText
    val tokensWithFlag = new SmartTextMapVectorizer()
      .setStripHtml(true).setInput(m1).tokenize(exampleHTML).tokens.value
    val tokensWithAnalyzer = new SmartTextMapVectorizer().setInput(m1)
      .tokenize(exampleHTML, analyzer = TextTokenizer.AnalyzerHtmlStrip).tokens.value
    tokensWithFlag should contain theSameElementsInOrderAs tokensWithAnalyzer
  }

  private[op] def assertVectorLength(df: DataFrame, output: FeatureLike[OPVector],
    expectedLength: Int, textVectorizationMethod: TextVectorizationMethod): Unit = {
    val result = df.collect(output)
    val firstRes = result.head
    val metaColumns = OpVectorMetadata(df.schema(output.name)).columns

    firstRes.v.size shouldBe expectedLength
    metaColumns.length shouldBe expectedLength

    textVectorizationMethod match {
      case Pivot => assert(metaColumns(expectedLength - 2).indicatorValue.contains(OpVectorColumnMetadata.OtherString))
      case Hash => assert(metaColumns(expectedLength - 2).descriptorValue
        .contains(OpVectorColumnMetadata.TextLenString))
      case other => throw new Error(s"Only Pivoting or Hashing possible, got ${other} instead")
    }
    assert(metaColumns(expectedLength - 1).indicatorValue.contains(OpVectorColumnMetadata.NullString))
  }

  /**
   * Set of tests to check that the derived quantities calculated on the length distribution in TextMapStats (for
   * a single key) match the actual length distributions of the tokens.
   *
   * @param res       TextMapStats result to compare
   * @param key       key to use for comparisons
   * @param lengthSeq Expected length sequence
   * @return          Assertions on derived quantities in TextStats
   */
  private[op] def checkDerivedQuantities(res: TextMapStats, key: String, lengthSeq: Seq[Long]): Assertion = {
    val expectedLengthMean = lengthSeq.sum.toDouble / lengthSeq.length
    val expectedLengthVariance = lengthSeq.map(x => math.pow((x - expectedLengthMean), 2)).sum / lengthSeq.length
    val expectedLengthStdDev = math.sqrt(expectedLengthVariance)

    res.keyValueCounts(key).lengthSize shouldBe lengthSeq.length
    compareWithTol(res.keyValueCounts(key).lengthMean, expectedLengthMean, tol)
    compareWithTol(res.keyValueCounts(key).lengthVariance, expectedLengthVariance, tol)
    compareWithTol(res.keyValueCounts(key).lengthStdDev, expectedLengthStdDev, tol)
  }
}
