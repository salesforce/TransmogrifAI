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
class TextTokenizerTest extends FlatSpec with TestSparkContext {

  // scalastyle:off
  val (ds, f1, f2, f3) = TestFeatureBuilder(
    Seq(
      ("I've got a lovely bunch of coconuts".toText, "古池や蛙飛び込む水の音".toText,
        "Première détection d’une atmosphère autour d’une exoplanète de la taille de la Terre".toText),
      ("There they are, all standing in a row".toText, "地磁気発生の謎に迫る地球内部の環境、再現実験".toText,
        "Les deux commissions, créées respectivement en juin 2016 et janvier 2017".toText),
      ("Big ones, small ones, some as big as your head".toText, "初めまして私はケビンです".toText,
        "Il publie sa théorie de la relativité restreinte en 1905".toText),
      ("".toText, Text.empty, Text.empty)
    )
  )
  // scalastyle:on

  val tokenizerEnglish = new TextTokenizer[Text]().setInput(f1).setDefaultLanguage(Language.English)
  val tokenizerAuto = new TextTokenizer[Text]().setAutoDetectLanguage(true)
  val tokenizerJapanese = new TextTokenizer[Text]().setDefaultLanguage(Language.Japanese).setInput(f2)
  val tokenizerFrench = new TextTokenizer[Text].setDefaultLanguage(Language.French).setInput(f3)

  classOf[TextTokenizer[_]].getSimpleName should "return single properly formed feature" in {
    val output1 = tokenizerEnglish.getOutput()
    val output2 = tokenizerJapanese.getOutput()
    val output3 = tokenizerFrench.getOutput()

    output1.name shouldBe tokenizerEnglish.outputName
    output1.parents shouldBe Array(f1)
    output1.originStage shouldBe tokenizerEnglish

    output2.name shouldBe tokenizerJapanese.outputName
    output2.parents shouldBe Array(f2)
    output2.originStage shouldBe tokenizerJapanese

    output3.name shouldBe tokenizerFrench.outputName
    output3.parents shouldBe Array(f3)
    output3.originStage shouldBe tokenizerFrench
  }

  it should "return empty TextList when input text is empty" in {
    val emptyInput = Text.empty

    tokenizerEnglish.transformFn(emptyInput) shouldBe TextList.empty
    tokenizerJapanese.transformFn(emptyInput) shouldBe TextList.empty
    tokenizerFrench.transformFn(emptyInput) shouldBe TextList.empty
  }

  it should "tokenize English text correctly" in {
    val transformed = tokenizerEnglish.transform(ds)
    val output = tokenizerEnglish.getOutput()
    val actualOutput = transformed.collect(output)

    actualOutput(0) shouldEqual List("i'v", "got", "love", "bunch", "coconut").toTextList
    actualOutput(1) shouldEqual List("all", "stand", "row").toTextList
    actualOutput(2) shouldEqual List("big", "on", "small", "on", "some", "big", "your", "head").toTextList
  }

  it should "tokenize Japanese text correctly" in {
    val transformed = tokenizerJapanese.transform(ds)
    val output = tokenizerJapanese.getOutput()
    val actualOutput = transformed.collect(output)

    // scalastyle:off
    actualOutput(0) shouldEqual List("古池", "蛙", "飛び込む", "水", "音").toTextList
    actualOutput(1) shouldEqual List("地磁気", "発生", "謎", "迫る", "地球", "内部", "環境", "再現", "実験").toTextList
    actualOutput(2) shouldEqual List("初め", "まして", "私", "ケビン").toTextList
    // scalastyle:on
  }

  it should "tokenize French text correctly" in {
    val transformed = tokenizerFrench.transform(ds)
    val output = tokenizerFrench.getOutput()
    val actualOutput = transformed.collect(output)

    actualOutput(0) shouldEqual List("premier", "detection", "atmosph", "autou", "exoplanet", "tail", "tere").toTextList
    actualOutput(1) shouldEqual List("deu", "comision", "cre", "respectif", "juin", "2016", "janvi", "2017").toTextList
    actualOutput(2) shouldEqual List("publ", "theo", "relativit", "restreint", "1905").toTextList
  }

  it should "auto detect languages and tokenize accordingly" in {
    val f1Out = tokenizerAuto.setInput(f1).transform(ds).collect(tokenizerAuto.getOutput())
    f1Out(0) shouldEqual List("i'v", "got", "love", "bunch", "coconut").toTextList
    f1Out(1) shouldEqual List("all", "stand", "row").toTextList
    f1Out(2) shouldEqual List("big", "on", "small", "on", "some", "big", "your", "head").toTextList

    val f2Out = tokenizerAuto.setInput(f2).transform(ds).collect(tokenizerAuto.getOutput())
    // scalastyle:off
    // Result 1 - is identified as Japanese and tokenized correctly
    f2Out(1) shouldEqual List("地磁気", "発生", "謎", "迫る", "地球", "内部", "環境", "再現", "実験").toTextList
    // Results 0 and 2 are miscategorized as Chinese, and since the confidence is low it falls back to english.
    f2Out(0) shouldEqual List("古", "池", "や", "蛙", "飛", "び", "込", "む", "水", "の", "音").toTextList
    f2Out(2) shouldEqual List("初", "め", "ま", "し", "て", "私", "は", "ケビン", "で", "す").toTextList
    // scalastyle:on

    val f3Out = tokenizerAuto.setInput(f3).transform(ds).collect(tokenizerAuto.getOutput())
    f3Out(0) shouldEqual List("premier", "detection", "atmosph", "autou", "exoplanet", "tail", "tere").toTextList
    f3Out(1) shouldEqual List("deu", "comision", "cre", "respectif", "juin", "2016", "janvi", "2017").toTextList
    f3Out(2) shouldEqual List("publ", "theo", "relativit", "restreint", "1905").toTextList
  }
  it should "auto detect languages and fall back to a default language correctly" in {
    val actualOutput =
      tokenizerAuto.setDefaultLanguage(Language.Japanese).setInput(f2).transform(ds).collect(tokenizerAuto.getOutput())
    // scalastyle:off
    actualOutput(0) shouldEqual List("古池", "蛙", "飛び込む", "水", "音").toTextList
    actualOutput(1) shouldEqual List("地磁気", "発生", "謎", "迫る", "地球", "内部", "環境", "再現", "実験").toTextList
    actualOutput(2) shouldEqual List("初め", "まして", "私", "ケビン").toTextList
    // scalastyle:on
  }

  it should "has a working shortcut" in {
    val tokenized = f1.tokenize()
    val actualOutput = tokenized.originStage.asInstanceOf[Transformer].transform(ds).collect(tokenized)

    actualOutput(0) shouldEqual List("i've", "got", "lovely", "bunch", "coconuts").toTextList
    actualOutput(1) shouldEqual List("all", "standing", "row").toTextList
    actualOutput(2) shouldEqual List("big", "ones", "small", "ones", "some", "big", "your", "head").toTextList
  }

  it should "filter out tokens based on min token length" in {
    val tokenized = f1.tokenize(minTokenLength = 5)
    val actualOutput = tokenized.originStage.asInstanceOf[Transformer].transform(ds).collect(tokenized)

    actualOutput(0) shouldEqual List("lovely", "bunch", "coconuts").toTextList
    actualOutput(1) shouldEqual List("standing").toTextList
    actualOutput(2) shouldEqual List("small").toTextList
  }

  it should "has a shortcut to tokenize using regex" in {
    the[IllegalArgumentException] thrownBy f1.tokenizeRegex(pattern = null)

    val tokenized = f1.tokenizeRegex(pattern = "\\s", minTokenLength = 5, toLowercase = false)
    val actualOutput = tokenized.originStage.asInstanceOf[Transformer].transform(ds).collect(tokenized)

    actualOutput(0) shouldEqual List("lovely", "bunch", "coconuts").toTextList
    actualOutput(1) shouldEqual List("There", "standing").toTextList
    actualOutput(2) shouldEqual List("ones,", "small", "ones,").toTextList
  }
}
