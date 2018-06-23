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

package com.salesforce.op.utils.text

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.feature.TextTokenizer
import com.salesforce.op.stages.impl.feature.TextTokenizer.TextTokenizerResult
import com.salesforce.op.test.TestCommon
import com.salesforce.op.utils.text.Language._
import opennlp.tools.sentdetect.SentenceModel
import opennlp.tools.tokenize.TokenizerModel
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpenNLPSentenceSplitterTest extends FlatSpec with TestCommon {

  val splitter = new OpenNLPSentenceSplitter()

  Spec[OpenNLPSentenceSplitter] should "split an English paragraph into sentences" in {
    val input =
      "Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov 29. " +
        "Mr Vinken is chairman of Elsevier N.V., the Dutch publishing group. Rudolph Agnew, 55 years old and " +
        "former chairman of Consolidated Gold Fields PLC, was named a director of this British industrial conglomerate."

    splitter.getSentences(input, language = English) shouldEqual Seq(
      "Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov 29.",
      "Mr Vinken is chairman of Elsevier N.V., the Dutch publishing group.",
      "Rudolph Agnew, 55 years old and former chairman of Consolidated Gold Fields PLC, " +
        "was named a director of this British industrial conglomerate."
    )

    TextTokenizer.tokenize(input.toText, sentenceSplitter = Option(splitter), defaultLanguage = English) shouldEqual
      TextTokenizerResult(English, Seq(
        Seq("pierr", "vinken", "61", "year", "old", "will", "join", "board",
          "nonexecut", "director", "nov", "29").toTextList,
        Seq("mr", "vinken", "chairman", "elsevi", "n.v", "dutch", "publish", "group").toTextList,
        Seq("rudolph", "agnew", "55", "year", "old", "former", "chairman", "consolid", "gold", "field", "plc",
          "name", "director", "british", "industri", "conglomer").toTextList))

    TextTokenizer.tokenize(input.toText, analyzer = new OpenNLPAnalyzer(), sentenceSplitter = Option(splitter),
      defaultLanguage = English) shouldEqual TextTokenizerResult(
      English, Seq(
        Seq("pierre", "vinken", ",", "61", "years", "old", ",", "will", "join", "the", "board", "as", "a",
          "nonexecutive", "director", "nov", "29", ".").toTextList,
        Seq("mr", "vinken", "is", "chairman", "of", "elsevier", "n", ".v.", ",", "the", "dutch", "publishing",
          "group", ".").toTextList,
        Seq("rudolph", "agnew", ",", "55", "years", "old", "and", "former", "chairman", "of", "consolidated",
          "gold", "fields", "plc", ",", "was", "named", "a", "director", "of", "this", "british", "industrial",
          "conglomerate", ".").toTextList))
  }

  it should "split a Portuguese text into sentences" in {
    // scalastyle:off
    val input = "Depois de Guimarães, o North Music Festival estaciona este ano no Porto. A partir de sexta-feira, " +
      "a Alfândega do Porto recebe a segunda edição deste festival de dois dias. No cartaz há nomes como os " +
      "portugueses Linda Martini e Mão Morta, mas também Guano Apes ou os DJ’s portugueses Rich e Mendes."

    splitter.getSentences(input, language = Portuguese) shouldEqual Seq(
      "Depois de Guimarães, o North Music Festival estaciona este ano no Porto.",
      "A partir de sexta-feira, a Alfândega do Porto recebe a segunda edição deste festival de dois dias.",
      "No cartaz há nomes como os portugueses Linda Martini e Mão Morta, mas também Guano Apes ou os DJ’s " +
        "portugueses Rich e Mendes."
    )
    // scalastyle:on
  }

  it should "load a sentence detection and tokenizer model for a language if they exist" in {
    val languages = Seq(Danish, Portuguese, English, Dutch, German, Sami)
    languages.map { language =>
      OpenNLPModels.getSentenceModel(language).exists(_.isInstanceOf[SentenceModel]) shouldBe true
      OpenNLPModels.getTokenizerModel(language).exists(_.isInstanceOf[TokenizerModel]) shouldBe true
    }
  }

  it should "load not a sentence detection and tokenizer model for a language if they do not exist" in {
    val languages = Seq(Japanese, Czech)
    languages.map { language =>
      OpenNLPModels.getSentenceModel(language) shouldEqual None
      OpenNLPModels.getTokenizerModel(language) shouldEqual None
    }
  }

  it should "return non-preprocessed input if no such a sentence detection model exist" in {
    // scalastyle:off
    val input = "ピエール・ヴィンケン（61歳）は、11月29日に臨時理事に就任します。" +
      "ヴィンケン氏は、オランダの出版グループであるエルゼビアN.V.の会長です。 " +
      "55歳のルドルフ・アグニュー（Rudolph Agnew、元コネチカットゴールドフィールドPLC）会長は、" +
      "この英国の産業大企業の取締役に任命されました。"
    // scalastyle:on
    splitter.getSentences(input, language = Language.Japanese) shouldEqual Seq(input)
  }
}
