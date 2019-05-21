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
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class TextNGramSimilarityTest extends OpTransformerSpec[RealNN, TextNGramSimilarity[Text]]{
  val(inputData, f1, f2) = TestFeatureBuilder(
    Seq[(Text, Text)](
      (Text("Hamlet: To be or not to be - that is the question."), Text("I like like Hamlet")),
      (Text("that is the question"), Text("There is no question")),
      (Text("Just some random text"), Text("I like like Hamlet")),
      (Text("Adobe CreativeSuite 5 Master Collection from cheap 4zp"),
        Text("Adobe CreativeSuite 5 Master Collection from cheap d1x")),
      (Text.empty, Text.empty),
      (Text(""), Text("")),
      (Text(""), Text.empty),
      (Text("asdf"), Text.empty),
      (Text.empty, Text("asdf"))
    )
  )

  val expectedResult = Seq(0.12666672468185425, 0.6083333492279053, 0.15873020887374878,
    0.9629629850387573, 0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  val nGramSimilarity = f1.toNGramSimilarity(f2, toLowerCase = false)
  val transformer = nGramSimilarity.originStage.asInstanceOf[TextNGramSimilarity[Text]]

  it should "correctly compute char-n-gram similarity with nondefault ngram param" in {
    val nGramSimilarity = f1.toNGramSimilarity(f2, nGramSize = 4, toLowerCase = false)
    val transformedDs = nGramSimilarity.originStage.asInstanceOf[Transformer].transform(inputData)
    val actualOutput = transformedDs.collect(nGramSimilarity)

    actualOutput shouldBe Seq(0.11500000953674316, 0.5666666626930237, 0.1547619104385376, 0.9722222089767456,
      0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }
}
