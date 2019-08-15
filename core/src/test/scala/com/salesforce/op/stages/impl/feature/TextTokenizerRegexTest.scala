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
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TextTokenizerRegexTest extends OpTransformerSpec[TextList, TextTokenizer[Text]] {

  val (inputData, english) = TestFeatureBuilder(
    Seq(
      "I've got a lovely bunch of coconuts",
      "There they are, all standing in a row",
      "Big ones, small ones, some as big as your head",
      "<body>Big ones, small <h1>ones</h1>, some as big as your head</body>",
      "two  words",
      "   ehh,   Fluff     ",
      ""
    ).map(_.toText)
  )

  val transformer =
    english.tokenizeRegex(pattern = "\\s+", minTokenLength = 5, toLowercase = false)
      .originStage.asInstanceOf[TextTokenizer[Text]]

  val expectedResult: Seq[TextList] = Array(
    List("lovely", "bunch", "coconuts").toTextList,
    List("There", "standing").toTextList,
    List("ones,", "small", "ones,").toTextList,
    List("<body>Big", "ones,", "small", "<h1>ones</h1>,", "head</body>").toTextList,
    List("words").toTextList,
    List("Fluff").toTextList,
    TextList.empty
  )
}
