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
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types.{Text, _}
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.SmartTextVectorizerAction._
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.stages.{NameDetectUtils, SensitiveFeatureMode}
import org.apache.log4j.Level
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class SmartTextMapVectorizerTest
  extends OpEstimatorSpec[OPVector, SequenceModel[TextMap, OPVector], SmartTextMapVectorizer[TextMap]]
    with AttributeAsserts {

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


  Spec[TextMapStats] should "provide a proper semigroup" in {
    val data = Seq(
      TextMapStats(Map(
        "f1" -> TextStats(Map("hello" -> 2, "world" -> 1)),
        "f2" -> TextStats(Map("hello" -> 2, "ocean" -> 2)),
        "f3" -> TextStats(Map("foo" -> 1))
      )),
      TextMapStats(Map(
        "f1" -> TextStats(Map("hello" -> 1)),
        "f2" -> TextStats(Map("ocean" -> 1, "other" -> 5))
      )),
      TextMapStats(Map(
        "f2" -> TextStats(Map("other" -> 1))
      ))
    )
    TextMapStats.monoid(2).sumOption(data) shouldBe Some(TextMapStats(Map(
      "f1" -> TextStats(Map("hello" -> 3, "world" -> 1)),
      "f2" -> TextStats(Map("hello" -> 2, "ocean" -> 3, "other" -> 5)),
      "f3" -> TextStats(Map("foo" -> 1))
    )))
  }

  it should "detect one categorical and one non-categorical text feature" in {
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

  /* TESTS FOR DETECTING SENSITIVE FEATURES BEGIN */
  lazy val (newInputData, features) = {
    val N = 5

    val baseText1 = Seq("hello world", "hello world", "good evening").toText ++ Seq(Text.empty, Text.empty)
    val baseText2 = Seq(
      "Hello world!", "What's up", "How are you doing, my friend?", "Not bad, my friend").toText :+ Text.empty
    val baseNames = Seq("Michael", "Michelle", "Roxanne", "Ross").toText :+ Text.empty

    def convertToMap(texts: Seq[Text]): Map[String, String] = texts.map(_.value).zipWithIndex.collect {
      case (Some(text), index) => (if (index == 3) "name" else s"text${index + 1}") -> text } toMap
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

    val nameTextMap: Seq[TextMap] = baseNames.map(_.value).collect { case Some(text) => TextMap(Map("name" -> text)) }
    val nameTextAreaMap: Seq[TextAreaMap] =
      baseNames.map(_.value).collect { case Some(text) => TextAreaMap(Map("name" -> text)) }

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
  newInputData.show(truncate = false)

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

  private lazy val NameDictionaryGroundTruth: RandomText[Text] = RandomText.textFromDomain(
    NameDetectUtils.DefaultNameDictionary.toList
  )

  it should "detect a single name feature" in {
    val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF7)
    val mapModel: SmartTextMapVectorizerModel[TextMap] = mapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextMap]]
    mapModel.args.allFeatureInfo.flatMap(_.map(_.whichAction)) shouldBe Seq(Sensitive)

    val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF8)
    val areaMapModel: SmartTextMapVectorizerModel[TextAreaMap] = areaMapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextAreaMap]]
    areaMapModel.args.allFeatureInfo.flatMap(_.map(_.whichAction)) shouldBe Seq(Sensitive)
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
    val (smart1, expected1) = result1.map(smartVector => smartVector -> OPVector.empty).unzip
    val (smart2, expected2) = result2.map(smartVector => smartVector -> OPVector.empty).unzip

    smart1 shouldBe expected1
    smart2 shouldBe expected2

    OpVectorMetadata("OutputVector", mapEstimator.getMetadata()).size shouldBe 0
    OpVectorMetadata("OutputVector", areaMapEstimator.getMetadata()).size shouldBe 0
  }

  it should "detect a single name column among other non-name Text columns" in {
    val mapEstimator: SmartTextMapVectorizer[TextMap] = biasMapEstimator.setInput(newF3, newF4)
    val mapModel: SmartTextMapVectorizerModel[TextMap] = mapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextMap]]
    println(mapModel.args.allFeatureInfo)
    mapModel.args.allFeatureInfo.flatMap(_.map(_.whichAction)) shouldBe
      Array(Categorical, NonCategorical, Sensitive)

    val areaMapEstimator: SmartTextMapVectorizer[TextAreaMap] = biasAreaMapEstimator.setInput(newF5, newF6)
    val areaMapModel: SmartTextMapVectorizerModel[TextAreaMap] = areaMapEstimator
      .fit(newInputData)
      .asInstanceOf[SmartTextMapVectorizerModel[TextAreaMap]]
    areaMapModel.args.allFeatureInfo.flatMap(_.map(_.whichAction)) shouldBe
      Array(Categorical, NonCategorical, Sensitive)
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

      OpVectorMetadata("OutputVector", mapEstimator.getMetadata()).size shouldBe
        OpVectorMetadata("OutputVector", withoutNamesEstimator.getMetadata()).size

      withNames shouldBe withoutNames
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
        OpVectorMetadata("OutputVector", oldMapEstimator.getMetadata()).size

      withNames shouldBe withoutNames
    }
  }

  it should "compute sensitive information in the metadata for one detected name column" in {
    loggingLevel(Level.DEBUG) // Changes SensitiveFeatureInformation creation logic
    def assertSensitive(estimator: SequenceEstimator[_, _], fname: String): Unit = {
      val sensitive = OpVectorMetadata("OutputVector", estimator.getMetadata()).sensitive
      println(sensitive)
      sensitive.get(fname) match {
        case Some(Seq(SensitiveFeatureInformation.Name(
          probName, genderDetectResults, probMale, probFemale, probOther, name, mapKey, actionTaken
        ))) =>
          probName shouldBe 1.0
          genderDetectResults.length shouldBe NameDetectUtils.GenderDetectStrategies.length
          probMale shouldBe 0.5
          probFemale shouldBe 0.5
          probOther shouldBe 0.0
          name shouldBe fname
          mapKey shouldBe "name"
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
  }
  /* TESTS FOR DETECTING SENSITIVE FEATURES END */
}
