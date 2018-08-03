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

import com.salesforce.op.features.types._
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.text.Language
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class NameEntityRecognizerTest extends OpTransformerSpec[MultiPickListMap, NameEntityRecognizer[Text]] {

  // Base tests
  val (inputData, inputText) = TestFeatureBuilder(Seq(
    ("Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29. Mr. Vinken is " +
      "chairman of Elsevier N.V., the Dutch publishing group. Rudolph Agnew, 55 years " +
      "old and former chairman of Consolidated Gold Fields PLC, was named a director of this " +
      "British industrial conglomerate.").toText))

  val transformer = new NameEntityRecognizer[Text].setInput(inputText)

  val expectedResult: Seq[MultiPickListMap] = Seq(
    Map("Rudolph" -> Set("Person"),
      "Agnew" -> Set("Person"),
      "Consolidated" -> Set("Organization"),
      "Vinken" -> Set("Person"),
      "Gold" -> Set("Organization"),
      "PLC" -> Set("Organization"),
      "Pierre" -> Set("Person"),
      "Fields" -> Set("Organization")
    ).toMultiPickListMap)

  it should "find the same set of name entities using the shortcut in RichTextFeatures" in {
    val nameEntityRecognizer = inputText.recognizeEntities().originStage.asInstanceOf[NameEntityRecognizer[Text]]
      .setInput(inputText)
    val transformed = nameEntityRecognizer.transform(inputData)
    val output = nameEntityRecognizer.getOutput()
    transformed.collect(output) shouldEqual expectedResult
  }

  it should "find name entities for Dutch text" in {
    // scalastyle:off
    val input = ("Pierre Vinken, 61 jaar oud, treedt toe tot het bestuur als een niet-uitvoerende " +
      "directeur op Nov. 29. De heer Vinken is voorzitter van Elsevier N.V., de Nederlandse uitgeversgroep. " +
      "Rudolph Agnew, 55 jaar oud en voormalig voorzitter van Consolidated Gold Fields PLC, werd benoemd tot " +
      "bestuurder van dit Britse industriële conglomeraat.").toText
    val expectedOutput = Map(
      "Nederlandse" -> Set("Misc"),
      "Nov." -> Set("Organization"),
      "Consolidated" -> Set("Misc"),
      "Vinken" -> Set("Person"),
      "Pierre" -> Set("Person"),
      "Britse" -> Set("Misc")
    ).toMultiPickListMap
    new NameEntityRecognizer[Text]().setDefaultLanguage(Language.Dutch).transformFn(input) shouldEqual expectedOutput
    // scalastyle:on
  }

  it should "return an empty map when there's no pre-trained name entity recognition model for the given language" in {
    val input = ("Pierre Vinken, mwenye umri wa miaka 61, atajiunga na bodi hiyo kama mkurugenzi asiyetarajiwa " +
      "Novemba 29. Mheshimiwa Vinken ni mwenyekiti wa Elsevier N.V., kundi la kuchapisha Kiholanzi. " +
      "Rudolph Agnew, mwenye umri wa miaka 55 na mwenyekiti wa zamani wa Mkutano Mkuu wa Gold Fields, " +
      "aliitwa mkurugenzi wa muungano huu wa viwanda wa Uingereza.").toText
    val expectedOutput = Map.empty[String, Set[String]].toMultiPickListMap
    new NameEntityRecognizer[Text]().setDefaultLanguage(Language.Swahili).transformFn(input) shouldEqual expectedOutput
  }

  // TODO: add a test for spanish NER after finding the spanish tokenizer
}
