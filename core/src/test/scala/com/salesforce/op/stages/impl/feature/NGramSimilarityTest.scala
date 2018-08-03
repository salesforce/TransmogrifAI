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
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class NGramSimilarityTest extends FlatSpec with TestSparkContext {

  val (dsCat, f1Cat, f2Cat) = TestFeatureBuilder(
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

  val(dsText, f1Text, f2Text) = TestFeatureBuilder(
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

  Spec[SetNGramSimilarity] should "correctly compute char-n-gram similarity" in {
    val catNGramSimilarity = f1Cat.toNGramSimilarity(f2Cat)
    val transformedDs = catNGramSimilarity.originStage.asInstanceOf[Transformer].transform(dsCat)
    val actualOutput = transformedDs.collect(catNGramSimilarity)

    actualOutput shouldBe Seq(0.3333333134651184, 0.09722214937210083, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }

  Spec[SetNGramSimilarity] should "correctly compute char-n-gram similarity with nondefault ngram param" in {
    val catNGramSimilarity = f1Cat.toNGramSimilarity(f2Cat, 5)
    val transformedDs = catNGramSimilarity.originStage.asInstanceOf[Transformer].transform(dsCat)
    val actualOutput = transformedDs.collect(catNGramSimilarity)

    actualOutput shouldBe Seq(0.3333333432674408, 0.12361115217208862, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }

  Spec[TextNGramSimilarity[_]] should "correctly compute char-n-gram similarity" in {
    val nGramSimilarity = f1Text.toNGramSimilarity(f2Text)
    val transformedDs = nGramSimilarity.originStage.asInstanceOf[Transformer].transform(dsText)
    val actualOutput = transformedDs.collect(nGramSimilarity)

    actualOutput shouldBe Seq(0.12666672468185425, 0.6083333492279053, 0.15873020887374878,
      0.9629629850387573, 0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }

  Spec[TextNGramSimilarity[_]] should "correctly compute char-n-gram similarity with nondefault ngram param" in {
    val nGramSimilarity = f1Text.toNGramSimilarity(f2Text, 4)
    val transformedDs = nGramSimilarity.originStage.asInstanceOf[Transformer].transform(dsText)
    val actualOutput = transformedDs.collect(nGramSimilarity)

    actualOutput shouldBe Seq(0.11500000953674316, 0.5666666626930237, 0.1547619104385376, 0.9722222089767456,
      0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }
}
