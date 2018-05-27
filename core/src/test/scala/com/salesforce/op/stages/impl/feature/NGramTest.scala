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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.NGram
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class NGramTest extends FlatSpec with TestSparkContext {

  val data = Seq("a b c d e f g").map(_.split(" ").toSeq.toTextList)
  lazy val (ds, f1) = TestFeatureBuilder(data)

  Spec[NGram] should "generate bigrams by default" in {
    val bigrams = f1.ngram()
    val transformedData = bigrams.originStage.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.collect(bigrams)

    bigrams.name shouldBe bigrams.originStage.getOutputFeatureName
    results(0) shouldBe Seq("a b", "b c", "c d", "d e", "e f", "f g").toTextList
  }

  it should "generate unigrams" in {
    val bigrams = f1.ngram(n = 1)
    val transformedData = bigrams.originStage.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.collect(bigrams)

    results(0) shouldBe data.head
  }

  it should "generate trigrams" in {
    val trigrams = f1.ngram(n = 3)
    val transformedData = trigrams.originStage.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.collect(trigrams)

    results(0) shouldBe Seq("a b c", "b c d", "c d e", "d e f", "e f g").toTextList
  }

  it should "not allow n < 1" in {
    the[IllegalArgumentException] thrownBy f1.ngram(n = 0)
    the[IllegalArgumentException] thrownBy f1.ngram(n = -1)
  }

}
