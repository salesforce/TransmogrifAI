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

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SmartTextMapVectorizerTest extends FlatSpec with TestSparkContext {
  lazy val (data, m1, m2, f1, f2) = TestFeatureBuilder("textMap1", "textMap2", "text1", "text2",
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
    TextMapStats.semiGroup(2).sumOption(data) shouldBe Some(TextMapStats(Map(
      "f1" -> TextStats(Map("hello" -> 3, "world" -> 1)),
      "f2" -> TextStats(Map("hello" -> 2, "ocean" -> 3, "other" -> 5)),
      "f3" -> TextStats(Map("foo" -> 1))
    )))
  }

  Spec[SmartTextMapVectorizer[_]] should "detect one categorical and one non-categorical text feature" in {
    val smartMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, smartVectorized).transform(data)
    val result = transformed.collect(smartMapVectorized, smartVectorized)

    val mapMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach{ case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      if (m.index < 4) m.indicatorGroup shouldBe f.indicatorGroup
      else m.indicatorGroup shouldBe Option(f2.name)
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach{ case (vec1, vec2) => vec1 shouldBe vec2}
  }

  it should "detect two categorical text features" in {
    val smartMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(10).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(10).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, smartVectorized).transform(data)
    val result = transformed.collect(smartMapVectorized, smartVectorized)

    val mapMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach{ case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      m.indicatorGroup shouldBe f.indicatorGroup
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach{ case (vec1, vec2) => vec1 shouldBe vec2}
  }

  it should "detect two non categorical text features" in {
    val smartMapVectorized = new SmartTextMapVectorizer[TextMap]()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(true)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, smartVectorized).transform(data)
    val result = transformed.collect(smartMapVectorized, smartVectorized)

    val mapMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach{ case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      if (m.index < 4) m.indicatorGroup shouldBe Option(f1.name)
      else m.indicatorGroup shouldBe Option(f2.name)
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach{ case (vec1, vec2) => vec1 shouldBe vec2}
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

    val transformed = new OpWorkflow().setResultFeatures(smartMapVectorized, shortcutMapVectorized).transform(data)
    val result = transformed.collect(smartMapVectorized, shortcutMapVectorized)

    val smartMeta = OpVectorMetadata(transformed.schema(smartMapVectorized.name))
    val shortcutMeta = OpVectorMetadata(transformed.schema(shortcutMapVectorized.name))
    smartMeta.history.keys shouldBe shortcutMeta.history.keys
    smartMeta.columns.length shouldBe shortcutMeta.columns.length

    smartMeta.columns.zip(shortcutMeta.columns).foreach{ case (smart, shortcut) =>
      smart.parentFeatureName shouldBe shortcut.parentFeatureName
      smart.parentFeatureType shouldBe shortcut.parentFeatureType
      smart.indicatorGroup shouldBe shortcut.indicatorGroup
      smart.indicatorValue shouldBe shortcut.indicatorValue
    }

    result.foreach{ case (vec1, vec2) => vec1 shouldBe vec2}
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

    val transformed = new OpWorkflow().setResultFeatures(textMapVectorized, textAreaMapVectorized).transform(data)
    val result = transformed.collect(textMapVectorized, textAreaMapVectorized)

    val textMapMeta = OpVectorMetadata(transformed.schema(textMapVectorized.name))
    val textareaMapMeta = OpVectorMetadata(transformed.schema(textAreaMapVectorized.name))
    textMapMeta.history.keys shouldBe textareaMapMeta.history.keys
    textMapMeta.columns.length shouldBe textareaMapMeta.columns.length

    textMapMeta.columns.zip(textareaMapMeta.columns).foreach{ case (textMap, textareaMap) =>
      textMap.parentFeatureName shouldBe textareaMap.parentFeatureName
      textMap.parentFeatureType shouldBe Array("com.salesforce.op.features.types.TextMap")
      textareaMap.parentFeatureType shouldBe Array("com.salesforce.op.features.types.TextAreaMap")
      textMap.indicatorGroup shouldBe textareaMap.indicatorGroup
      textMap.indicatorValue shouldBe textareaMap.indicatorValue
    }

    result.foreach{ case (vec1, vec2) => vec1 shouldBe vec2}
  }
}
