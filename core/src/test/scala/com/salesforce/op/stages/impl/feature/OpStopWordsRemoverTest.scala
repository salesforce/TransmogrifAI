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
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.test.{SwTransformerSpec, TestFeatureBuilder}
import org.apache.spark.ml.feature.StopWordsRemover
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpStopWordsRemoverTest extends SwTransformerSpec[TextList, StopWordsRemover, OpStopWordsRemover] {
  val data = Seq(
    "I AM groot", "Groot call me human", "or I will crush you"
  ).map(_.split(" ").toSeq.toTextList)

  val (inputData, textListFeature) = TestFeatureBuilder(data)

  val bigrams = textListFeature.removeStopWords()
  val transformer = bigrams.originStage.asInstanceOf[OpStopWordsRemover]

  val expectedResult = Seq(Seq("groot"), Seq("Groot", "call", "human"), Seq("crush")).map(_.toTextList)

  it should "allow case sensitivity" in {
    val noStopWords = textListFeature.removeStopWords(caseSensitive = true)
    val res = noStopWords.originStage.asInstanceOf[OpStopWordsRemover].transform(inputData)
    res.collect(noStopWords) shouldBe Seq(
      Seq("I", "AM", "groot"), Seq("Groot", "call", "human"), Seq("I", "crush")).map(_.toTextList)
  }

  it should "set custom stop words" in {
    val noStopWords = textListFeature.removeStopWords(stopWords = Array("Groot", "I"))
    val res = noStopWords.originStage.asInstanceOf[OpStopWordsRemover].transform(inputData)
    res.collect(noStopWords) shouldBe Seq(
      Seq("AM"), Seq("call", "me", "human"), Seq("or", "will", "crush", "you")).map(_.toTextList)
  }
}
