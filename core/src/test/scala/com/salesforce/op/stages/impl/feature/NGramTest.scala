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
import com.salesforce.op.stages.sparkwrappers.specific.OpTransformerWrapper
import com.salesforce.op.test.{SwTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.NGram
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class NGramTest extends SwTransformerSpec[TextList, NGram, OpTransformerWrapper[TextList, TextList, NGram]] {
  val data = Seq("a b c d e f g").map(_.split(" ").toSeq.toTextList)
  val (inputData, textListFeature) = TestFeatureBuilder(data)

  val expectedResult = Seq(Seq("a b", "b c", "c d", "d e", "e f", "f g").toTextList)

  val bigrams = textListFeature.ngram()
  val transformer = bigrams.originStage.asInstanceOf[OpTransformerWrapper[TextList, TextList, NGram]]

  it should "generate unigrams" in {
    val unigrams = textListFeature.ngram(n = 1)
    val transformedData = unigrams.originStage.asInstanceOf[Transformer].transform(inputData)
    val results = transformedData.collect(unigrams)

    results(0) shouldBe data.head
  }

  it should "generate trigrams" in {
    val trigrams = textListFeature.ngram(n = 3)
    val transformedData = trigrams.originStage.asInstanceOf[Transformer].transform(inputData)
    val results = transformedData.collect(trigrams)

    results(0) shouldBe Seq("a b c", "b c d", "c d e", "d e f", "e f g").toTextList
  }

  it should "not allow n < 1" in {
    the[IllegalArgumentException] thrownBy textListFeature.ngram(n = 0)
    the[IllegalArgumentException] thrownBy textListFeature.ngram(n = -1)
  }

}