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

package com.salesforce.op.dsl

import com.salesforce.op.features.Feature
import com.salesforce.op.features.types.TextList
import com.salesforce.op.test.{FeatureTestBase, TestFeatureBuilder, TestSparkContext}
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RichListFeatureTest extends FlatSpec with FeatureTestBase with RichFeature  with TestSparkContext{
  val testData = Seq(
    "To be, or not to be, that is the question:",
    "Whether 'tis nobler in the mind to suffer",
    "The slings and arrows of outrageous fortune",
    "Or to take Arms against a Sea of troubles,"
  ).map(_.toLowerCase.split(" ").toSeq).map( x => TextList(x))


  val (ds, f1): (DataFrame, Feature[TextList]) = TestFeatureBuilder(testData)
  Spec[RichListFeature] should "return a uid and name for a tf" in {
    val tf_value = f1.tf()
    val all_features = tf_value.allFeatures
    val uid = tf_value.uid
    val name = tf_value.name
    uid shouldBe "OPVector_000000000002"
    name shouldBe "f1_1-stagesApplied_OPVector_000000000002"
  }

  it should "return a name and uid for tfidf" in {
    val tfIdfValues = f1.tfidf()
    val name = tfIdfValues.name
    val uid = tfIdfValues.uid
    uid shouldBe "OPVector_000000000004"
    name shouldBe "f1_2-stagesApplied_OPVector_000000000004"
  }

  it should "return a feature length for tfidf function" in {
    val tfIdfValues = f1.tfidf()
    val feature_length = tfIdfValues.allFeatures.length
    feature_length shouldBe 3
  }

  it should "return non null word2vec" in {
    val word2vec = f1.word2vec()
    val word2vecUid = word2vec.uid
    word2vecUid shouldBe "OPVector_000000000007"
  }
}
