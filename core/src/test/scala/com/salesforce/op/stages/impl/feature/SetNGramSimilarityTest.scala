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
class SetNGramSimilarityTest extends OpTransformerSpec[RealNN, NGramSimilarity[MultiPickList]] {

  val (inputData, f1, f2) = TestFeatureBuilder(
    Seq(
      (Seq("Red", "Green"), Seq("Red")),
      (Seq("Red", "Green"), Seq("Yellow, Blue")),
      (Seq("Red", "Yellow"), Seq("Red", "Yellow")),
      (Seq[String](), Seq("Red", "Yellow")),
      (Seq[String](), Seq[String]()),
      (Seq[String](""), Seq[String]("asdf")),
      (Seq[String](""), Seq[String]("")),
      (Seq[String]("", ""), Seq[String]("", ""))
    ).map(v => v._1.toMultiPickList -> v._2.toMultiPickList)
  )

  val expectedResult = Seq(0.3333333134651184, 0.09722214937210083, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  val catNGramSimilarity = f1.toNGramSimilarity(f2)
  val transformer = catNGramSimilarity.originStage.asInstanceOf[NGramSimilarity[MultiPickList]]

  it should "correctly compute char-n-gram similarity with nondefault ngram param" in {
    val cat5GramSimilarity = f1.toNGramSimilarity(f2, 5)
    val transformedDs = cat5GramSimilarity.originStage.asInstanceOf[Transformer].transform(inputData)
    val actualOutput = transformedDs.collect(cat5GramSimilarity)

    actualOutput shouldBe Seq(0.3333333432674408, 0.12361115217208862, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }
}

