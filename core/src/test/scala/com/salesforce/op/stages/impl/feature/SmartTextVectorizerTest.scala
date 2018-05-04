/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
class SmartTextVectorizerTest extends FlatSpec with TestSparkContext {
  lazy val (data, f1, f2) = TestFeatureBuilder("text1", "text2",
    Seq[(Text, Text)](
      ("hello world".toText, "Hello world!".toText),
      ("hello world".toText, "What's up".toText),
      ("good evening".toText, "How are you doing, my friend?".toText),
      ("hello world".toText, "Not bad, my friend.".toText),
      (Text.empty, Text.empty)
    )
  )

  Spec[TextStats] should "aggregate correctly" in {
    val l1 = TextStats(Map("hello" -> 1, "world" -> 2))
    val r1 = TextStats(Map("hello" -> 1, "world" -> 1))
    val expected1 = TextStats(Map("hello" -> 2, "world" -> 3))

    val l2 = TextStats(Map("hello" -> 1, "world" -> 2, "ocean" -> 3))
    val r2 = TextStats(Map("hello" -> 1))
    val expected2 = TextStats(Map("hello" -> 1, "world" -> 2, "ocean" -> 3))

    TextStats.semiGroup(2).plus(l1, r1) shouldBe expected1
    TextStats.semiGroup(2).plus(l2, r2) shouldBe expected2
  }

  Spec[SmartTextVectorizer[_]] should "detect one categorical and one non-categorical text feature" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val categoricalVectorized = new OpTextPivotVectorizer[Text]().setMinSupport(1).setTopK(2).setInput(f1).getOutput()
    val tokenizedText = new TextTokenizer[Text]().setInput(f2).getOutput()
    val textVectorized = new OPCollectionHashingVectorizer[TextList]()
      .setNumFeatures(4).setPrependFeatureName(false).setInput(tokenizedText).getOutput()
    val nullIndicator = new TextListNullTransformer[TextList]().setInput(tokenizedText).getOutput()

    val transformed = new OpWorkflow()
      .setResultFeatures(smartVectorized, categoricalVectorized, textVectorized, nullIndicator).transform(data)
    val result = transformed.collect(smartVectorized, categoricalVectorized, textVectorized, nullIndicator)

    val (smart, expected) = result.map{ case (smartVector, categoricalVector, textVector, nullVector) =>
      val combined = VectorsCombiner.combineOP(Seq(categoricalVector, textVector, nullVector))
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

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized, categoricalVectorized).transform(data)
    val result = transformed.collect(smartVectorized, categoricalVectorized)

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

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized, textVectorized, nullIndicator).transform(data)
    val result = transformed.collect(smartVectorized, textVectorized, nullIndicator)

    val (smart, expected) = result.map{ case (smartVector, textVector, nullVector) =>
      val combined = VectorsCombiner.combineOP(Seq(textVector, nullVector))
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
      autoDetectLanguage = false, minTokenLength = 1, toLowercase = false, forceSharedHashSpace = true,
      others = Array(f2)
    )

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized, shortcutVectorized).transform(data)
    val result = transformed.collect(smartVectorized, shortcutVectorized)

    val (regular, shortcut) = result.unzip

    regular shouldBe shortcut
  }

  it should "fail with an assertion error" in {
    val emptyDF = data.filter(data("text1") === "").toDF()

    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val thrown = intercept[AssertionError] {
      new OpWorkflow().setResultFeatures(smartVectorized).transform(emptyDF)
    }
    assert(thrown.getMessage.contains("assertion failed"))
  }

  it should "generate metadata correctly" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(2).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(data)

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 9
    meta.columns.foreach{ col =>
      if (col.index < 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.indicatorGroup shouldBe Option(f1.name)
      } else if (col.index == 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.indicatorGroup shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(TransmogrifierDefaults.OtherString)
      } else if (col.index == 3) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.indicatorGroup shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.indicatorGroup shouldBe None
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.indicatorGroup shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "generate categorical metadata correctly" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(4).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(data)

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 8
    meta.columns.foreach{ col =>
      if (col.index < 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.indicatorGroup shouldBe Option(f1.name)
      } else if (col.index == 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.indicatorGroup shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(TransmogrifierDefaults.OtherString)
      } else if (col.index == 3) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.indicatorGroup shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else if (col.index < 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.indicatorGroup shouldBe Option(f2.name)
      } else if (col.index == 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.indicatorGroup shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(TransmogrifierDefaults.OtherString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.indicatorGroup shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "generate non categorical metadata correctly" in {
    val smartVectorized = new SmartTextVectorizer()
      .setMaxCardinality(1).setNumFeatures(4).setMinSupport(1).setTopK(2).setPrependFeatureName(false)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(smartVectorized).transform(data)

    val meta = OpVectorMetadata(transformed.schema(smartVectorized.name))
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 10
    meta.columns.foreach{ col =>
      if (col.index < 4) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.indicatorGroup shouldBe None
      } else if (col.index < 8) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.indicatorGroup shouldBe None
      } else if (col.index == 8) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.indicatorGroup shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.indicatorGroup shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

}
