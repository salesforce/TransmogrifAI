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

package com.salesforce.op.utils.text

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.feature.NameEntityRecognizer
import com.salesforce.op.test.TestCommon
import com.salesforce.op.utils.text.NameEntityType._
import opennlp.tools.util.Span
import org.junit.runner.RunWith
import org.scalatest._
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpenNLPNameEntityTaggerTest extends FlatSpec with TestCommon {

  val nerTagger = new OpenNLPNameEntityTagger()

  Spec[OpenNLPNameEntityTagger] should "return the consistent results as expected" in {
    val input = Seq(
      "Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.",
      "Rudolph Agnew, 55 years old and former chairman of Consolidated Gold Fields PLC, was named a director of this" +
        "a director of this British industrial conglomerate."
    )
    val tokens: Seq[TextList] = input.map(x => NameEntityRecognizer.Analyzer.analyze(x, Language.English).toTextList)
    val expectedOutputs = Seq(
      Map("Vinken" -> Set(Person), "Pierre" -> Set(Person)),
      Map("Agnew" -> Set(Person), "Rudolph" -> Set(Person))
    )
    tokens.zip(expectedOutputs).foreach { case (tokenInput, expected) =>
      nerTagger.tag(tokenInput.value, Language.English, Seq(NameEntityType.Person)).tokenTags shouldEqual expected
    }
  }

  it should "load all the existing name entity recognition models" in {
    val languageNameEntityPairs = Seq(
      (Language.English, NameEntityType.Date),
      (Language.English, NameEntityType.Location),
      (Language.English, NameEntityType.Money),
      (Language.English, NameEntityType.Organization),
      (Language.English, NameEntityType.Percentage),
      (Language.English, NameEntityType.Person),
      (Language.English, NameEntityType.Time),
      (Language.Spanish, NameEntityType.Location),
      (Language.Spanish, NameEntityType.Organization),
      (Language.Spanish, NameEntityType.Person),
      (Language.Spanish, NameEntityType.Misc),
      (Language.Dutch, NameEntityType.Location),
      (Language.Dutch, NameEntityType.Organization),
      (Language.Dutch, NameEntityType.Person),
      (Language.Dutch, NameEntityType.Misc)
    )
    languageNameEntityPairs.foreach { case (l, n) =>
      OpenNLPModels.getTokenNameFinderModel(l, n).isDefined shouldBe true
    }
  }

  it should "not get any model correctly if no such model exists" in {
    val languageNameEntityPairs = Seq(
      (Language.Unknown, NameEntityType.Other),
      (Language.Urdu, NameEntityType.Location)
    )
    languageNameEntityPairs.foreach { case (l, n) =>
      OpenNLPModels.getTokenNameFinderModel(l, n) shouldBe None
    }
  }

  // test the convertSpansToMap function
  it should "retrieve correct information from the output of name entity recognition model" in {
    val inputs = Seq(Array("ab", "xx", "yy", "zz", "ss", "dd", "cc") ->
      Seq(new Span(2, 4, "person"), new Span(3, 5, "location")), // interweaving entities
      Array("a", "b", "c", "d") -> Seq(new Span(3, 4, "location")), // end of sentence entity
      Array("a", "b", "c", "d") -> Seq(new Span(0, 2, "location")), // beginning of sentence entity
      Array("a", "b", "c", "d") -> Seq.empty
    )
    val expectedOutputs = Seq(
      Map("yy" -> Set(Person), "zz" -> Set(Person, Location), "ss" -> Set(Location)),
      Map("d" -> Set(Location)),
      Map("a" -> Set(Location), "b" -> Set(Location)),
      Map.empty[String, Set[String]]
    )

    inputs.zip(expectedOutputs).map { case (tokensInput, expected) =>
      val actual = nerTagger.convertSpansToMap(tokensInput._2, tokensInput._1)
      actual shouldEqual expected
    }
  }

}
