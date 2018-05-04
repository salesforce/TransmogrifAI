/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.text.Language
import org.apache.spark.ml.Transformer
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class LangDetectorTest extends FlatSpec with TestSparkContext {

  // scalastyle:off
  val (ds, f1, f2, f3) = TestFeatureBuilder(
    Seq(
      (
        "I've got a lovely bunch of coconuts".toText,
        "文化庁によりますと、世界文化遺産への登録を目指している、福岡県の「宗像・沖ノ島と関連遺産群」について、ユネスコの諮問機関は、８つの構成資産のうち、沖ノ島など４つについて、「世界遺産に登録することがふさわしい」とする勧告をまとめました。".toText,
        "Première détection d’une atmosphère autour d’une exoplanète de la taille de la Terre".toText
      ),
      (
        "There they are, all standing in a row".toText,
        "地磁気発生の謎に迫る地球内部の環境、再現実験".toText,
        "Les deux commissions, créées respectivement en juin 2016 et janvier 2017".toText
      ),
      (
        "Big ones, small ones, some as big as your head".toText,
        "大学レスリング界で「黒船」と呼ばれたカザフスタン出身の大型レスラーが、日本の男子グレコローマンスタイルの重量級強化のために一役買っている。山梨学院大をこの春卒業したオレッグ・ボルチン（２４）。４月から新日本プロレスの親会社ブシロードに就職。自身も日本を拠点に、アマチュアレスリングで２０２０年東京五輪を目指す。".toText,
        "Il publie sa théorie de la relativité restreinte en 1905".toText
      )
    )
  )
  // scalastyle:on
  val langDetector = new LangDetector[Text]().setInput(f1)

  classOf[LangDetector[_]].getSimpleName should "return single properly formed feature" in {
    val output1 = langDetector.getOutput()

    output1.name shouldBe langDetector.getOutputFeatureName
    output1.parents shouldBe Array(f1)
    output1.originStage shouldBe langDetector
  }

  it should "return empty RealMap when input text is empty" in {
    langDetector.transformFn(Text.empty) shouldBe RealMap.empty
  }

  it should "detect English language" in {
    assertDetectionResults(
      results = langDetector.setInput(f1).transform(ds).collect(langDetector.getOutput()),
      expectedLanguage = Language.English
    )
  }

  it should "detect Japanese language" in {
    assertDetectionResults(
      results = langDetector.setInput(f2).transform(ds).collect(langDetector.getOutput()),
      expectedLanguage = Language.Japanese
    )
  }

  it should "detect French language" in {
    assertDetectionResults(
      results = langDetector.setInput(f3).transform(ds).collect(langDetector.getOutput()),
      expectedLanguage = Language.French
    )
  }

  it should "has a working shortcut" in {
    val tokenized = f1.detectLanguages()

    assertDetectionResults(
      results = tokenized.originStage.asInstanceOf[Transformer].transform(ds).collect(tokenized),
      expectedLanguage = Language.English
    )
  }

  private def assertDetectionResults
  (
    results: Array[RealMap],
    expectedLanguage: Language,
    confidence: Double = 0.99
  ): Unit =
    results.foreach(res => {
      res.value.size shouldBe 1
      res.value.contains(expectedLanguage.entryName) shouldBe true
      res.value(expectedLanguage.entryName) should be >= confidence
    })

}
