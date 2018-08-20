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

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.text.Language
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TextTokenizerTest extends FlatSpec with TestSparkContext {

  // scalastyle:off
  val (data, english, japanese, french) = TestFeatureBuilder(
    Seq(
      ("I've got a lovely bunch of coconuts".toText,
        "古池や蛙飛び込む水の音".toText,
        "Première détection d’une atmosphère autour d’une exoplanète de la taille de la Terre".toText
      ),
      ("There they are, all standing in a row".toText,
        "地磁気発生の謎に迫る地球内部の環境、再現実験".toText,
        "Les deux commissions, créées respectivement en juin 2016 et janvier 2017".toText
      ),
      ("Big ones, small ones, some as big as your head".toText,
        "初めまして私はケビンです".toText,
        "Il publie sa théorie de la relativité restreinte en 1905".toText
      ),
      ("<body>Big ones, small <h1>ones</h1>, some as big as your head</body>".toText,
        "初めまして私はケビンです, <h1>初めまして私はケビンです</h1>".toText,
        "Il <h2 class=\"a\">publie sa théorie de la relativité restreinte en 1905".toText
      ),
      ("".toText, Text.empty, Text.empty)
    )
  )
  // scalastyle:on

  trait English {
    val expected = Array(
      List("got", "love", "bunch", "coconut").toTextList,
      List("stand", "row").toTextList,
      List("big", "on", "small", "on", "big", "head").toTextList,
      List("bodi", "big", "on", "small", "h1", "on", "h1", "big", "head", "bodi").toTextList,
      TextList.empty
    )
    val expectedHtml = {
      val copy = expected.toList.toArray
      copy(3) = List("big", "on", "small", "on", "big", "head").toTextList
      copy
    }
  }
  trait Japanese {
    // scalastyle:off
    val expected = Array(
      List("古池", "蛙", "飛び込む", "水", "音").toTextList,
      List("地磁気", "発生", "謎", "迫る", "地球", "内部", "環境", "再現", "実験").toTextList,
      List("初め", "まして", "私", "ケビン").toTextList,
      List("初め", "まして", "私", "ケビン", "h", "1", "初め", "まして", "私", "ケビン", "h", "1").toTextList,
      TextList.empty
    )
    val expectedHtml = {
      val copy = expected.toList.toArray
      copy(3) = List("初め", "まして", "私", "ケビン", "初め", "まして", "私", "ケビン").toTextList
      copy
    }
    val expectedAuto = Array(
      // Results 0, 2, 3 are miscategorized as Chinese, and since the confidence is low it falls back to english.
      // Result 1 - is identified as Japanese and tokenized correctly
      List("古", "池", "や", "蛙", "飛", "び", "込", "む", "水", "の", "音").toTextList,
      List("地磁気", "発生", "謎", "迫る", "地球", "内部", "環境", "再現", "実験").toTextList,
      List("初", "め", "ま", "し", "て", "私", "は", "ケビン", "で", "す").toTextList,
      List("初", "め", "ま", "し", "て", "私", "は", "ケビン", "で", "す", "h1",
        "初", "め", "ま", "し", "て", "私", "は", "ケビン", "で", "す", "h1").toTextList,
      TextList.empty
    )
    // scalastyle:on
  }
  trait French {
    // scalastyle:off
    val expected = Array(
      List("premier", "detection", "atmosph", "autou", "exoplanet", "tail", "tere").toTextList,
      List("deu", "comision", "cre", "respectif", "juin", "2016", "janvi", "2017").toTextList,
      List("publ", "theo", "relativit", "restreint", "1905").toTextList,
      List("h2", "clas", "a", "publ", "theo", "relativit", "restreint", "1905").toTextList,
      TextList.empty
    )
    val expectedHtml = {
      val copy = expected.toList.toArray
      copy(3) = List("publ", "theo", "relativit", "restreint", "1905").toTextList
      copy
    }
    // scalastyle:on
  }

  Spec[TextTokenizer[_]] should "tokenize text correctly [English]" in new English {
    assertTextTokenizer(
      input = english, expected = expected,
      tokenizer = new TextTokenizer[Text]().setDefaultLanguage(Language.English)
    )
  }
  it should "tokenize text correctly [Japanese]" in new Japanese {
    assertTextTokenizer(
      input = japanese, expected = expected,
      tokenizer = new TextTokenizer[Text]().setDefaultLanguage(Language.Japanese)
    )
  }
  it should "tokenize text correctly [French]" in new French {
    assertTextTokenizer(
      input = french, expected = expected,
      tokenizer = new TextTokenizer[Text]().setDefaultLanguage(Language.French)
    )
  }
  it should "strip html tags and tokenize text correctly [English]" in new English {
    assertTextTokenizer(
      input = english, expected = expectedHtml,
      tokenizer = new TextTokenizer[Text](analyzer = TextTokenizer.AnalyzerHtmlStrip)
        .setDefaultLanguage(Language.English)
    )
  }
  it should "strip html tags and tokenize text correctly [Japanese]" in new Japanese {
    assertTextTokenizer(
      input = japanese, expected = expectedHtml,
      tokenizer = new TextTokenizer[Text](analyzer = TextTokenizer.AnalyzerHtmlStrip)
        .setDefaultLanguage(Language.Japanese)
    )
  }
  it should "strip html tags and tokenize text correctly [French]" in new French {
    assertTextTokenizer(
      input = french, expected = expectedHtml,
      tokenizer = new TextTokenizer[Text](analyzer = TextTokenizer.AnalyzerHtmlStrip)
        .setDefaultLanguage(Language.French)
    )
  }
  it should "auto detect languages and tokenize accordingly [English]" in new English {
    assertTextTokenizer(
      input = english, expected = expected,
      tokenizer = new TextTokenizer[Text]().setAutoDetectLanguage(true)
    )
  }
  it should "auto detect languages and tokenize accordingly [Japanese]" in new Japanese {
    assertTextTokenizer(
      input = japanese, expected = expectedAuto,
      tokenizer = new TextTokenizer[Text]().setAutoDetectLanguage(true)
    )
  }
  it should "auto detect languages and tokenize accordingly [French]" in new French {
    assertTextTokenizer(
      input = french, expected = expected,
      tokenizer = new TextTokenizer[Text]().setAutoDetectLanguage(true)
    )
  }
  it should "work as a shortcut" in {
    val tokenized = english.tokenize()
    assertTextTokenizer(
      input = english,
      tokenizer = tokenized.originStage.asInstanceOf[TextTokenizer[Text]],
      expected = Array(
        List("got", "lovely", "bunch", "coconuts").toTextList,
        List("standing", "row").toTextList,
        List("big", "ones", "small", "ones", "big", "head").toTextList,
        List("body", "big", "ones", "small", "h1", "ones", "h1", "big", "head", "body").toTextList,
        TextList.empty
      )
    )
  }
  it should "filter out tokens based on min token length" in {
    val tokenized = english.tokenize(minTokenLength = 5)
    assertTextTokenizer(
      input = english,
      tokenizer = tokenized.originStage.asInstanceOf[TextTokenizer[Text]],
      expected = Array(
        List("lovely", "bunch", "coconuts").toTextList,
        List("standing").toTextList,
        List("small").toTextList,
        List("small").toTextList,
        TextList.empty
      )
    )
  }
  it should "has a shortcut to tokenize using regex" in {
    val tokenized = english.tokenizeRegex(pattern = "\\s", minTokenLength = 5, toLowercase = false)
    assertTextTokenizer(
      input = english,
      tokenizer = tokenized.originStage.asInstanceOf[TextTokenizer[Text]],
      expected = Array(
        List("lovely", "bunch", "coconuts").toTextList,
        List("There", "standing").toTextList,
        List("ones,", "small", "ones,").toTextList,
        List("<body>Big", "ones,", "small", "<h1>ones</h1>,", "head</body>").toTextList,
        TextList.empty
      )
    )
  }

  private def assertTextTokenizer(
    input: FeatureLike[Text],
    tokenizer: TextTokenizer[Text],
    expected: Array[TextList]
  ) = {
    val output = tokenizer.setInput(input).getOutput()
    output.name shouldBe tokenizer.getOutputFeatureName
    output.parents shouldBe Array(input)
    output.originStage shouldBe tokenizer

    val emptyInput = Text.empty
    tokenizer.transformFn(emptyInput) shouldBe TextList.empty

    val transformed = tokenizer.transform(data)
    val actualOutput = transformed.collect(output)
    actualOutput should contain theSameElementsInOrderAs expected
  }

}
