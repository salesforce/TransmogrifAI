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

import com.salesforce.op.OpWorkflow
import com.salesforce.op.dsl.{RichFeature, RichMapFeature, RichTextFeature}
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.test.{FeatureTestBase, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class EmailVectorizerTest
  extends FlatSpec with FeatureTestBase with RichMapFeature with RichFeature with RichTextFeature {
  val emailKey = "Email1"
  val emailKey2 = "Email2"
  val emails = (RandomText.emails("salesforce.com").take(2) ++ RandomText.emails("einstein.ai").take(2)).toSeq
  val emails2 = (RandomText.emails("einstein.ai").take(2) ++ RandomText.emails("salesforce.com").take(2)).toSeq
  val emailMap = emails.zip(emails2)
    .map { case (one, two) => Map(emailKey -> one.value.get, emailKey2 -> two.value.get).toEmailMap }

  private val CleanText = true
  private val CleanKeys = true
  private val MinSupport = 0
  private val TopK = 10

  val expectedEmailMap = Array(
    Vectors.dense(1.0, 0.0, 0.0),
    Vectors.dense(1.0, 0.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0)
  ).map(_.toOPVector)

  val expectedEmail = Array(
    Vectors.dense(1.0, 0.0, 0.0, 0.0),
    Vectors.dense(1.0, 0.0, 0.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0, 0.0)
  ).map(_.toOPVector)

  val expectedTrackNulls = Array(
    Vectors.dense(0.0, 1.0, 0.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0, 0.0),
    Vectors.dense(1.0, 0.0, 0.0, 0.0),
    Vectors.dense(1.0, 0.0, 0.0, 0.0)
  ).map(_.toOPVector)


  def transformAndCollect(ds: DataFrame, feature: FeatureLike[OPVector]): Array[OPVector] = {
    val transformed = new OpWorkflow().setResultFeatures(feature).transform(ds)
    val field = transformed.schema(feature.name)
    val collected = transformed.collect(feature)
    AttributeTestUtils.assertNominal(field, Array.fill(collected.head.value.size)(true))
    collected
  }

  Spec[RichEmailMapFeature] should "vectorize EmailMaps correctly" in {
    val (ds1, f1) = TestFeatureBuilder(emails.map(e => Map(emailKey -> e.value.get).toEmailMap))
    val vectorized = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, trackNulls = false)
    vectorized.originStage shouldBe a[TextMapPivotVectorizer[_]]
    vectorized shouldBe a[FeatureLike[_]]

    val result = transformAndCollect(ds1, vectorized)
    result(0) shouldBe result(1)
    result(2) shouldBe result(3)
    result should contain theSameElementsAs expectedEmailMap
  }

  it should "track nulls" in {
    val (ds1, f1) = TestFeatureBuilder(emails.map(e => Map(emailKey -> e.value.get).toEmailMap))
    val vectorized = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, trackNulls = true)
    vectorized.originStage shouldBe a[TextMapPivotVectorizer[_]]
    vectorized shouldBe a[FeatureLike[_]]

    val result = transformAndCollect(ds1, vectorized)
    result(0) shouldBe result(1)
    result(2) shouldBe result(3)
    result should contain theSameElementsAs expectedTrackNulls
  }

  it should "work on multiple keys in EmailMap" in {
    val expectedMulti = Array(
      Vectors.dense(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )

    val (ds1, f1) = TestFeatureBuilder(emailMap)
    val vectorized = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, trackNulls = false)

    val result = transformAndCollect(ds1, vectorized).map(_.value.toDense)
    result shouldBe expectedMulti
  }

  it should "track  nulls with multiple keys in EmailMap" in {
    val expectedMulti = Array(
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    )

    val (ds1, f1) = TestFeatureBuilder(emailMap)
    val vectorized = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, trackNulls = true)

    val result = transformAndCollect(ds1, vectorized).map(_.value.toDense)
    result shouldBe expectedMulti
  }

  it should "use whitelisted/ignore blacklisted keys in EmailMap" in {
    val (ds1, f1) = TestFeatureBuilder(emailMap)
    val vectorized = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, blackListKeys = Array(emailKey2), trackNulls = false)

    val result = transformAndCollect(ds1, vectorized)
    result(0) shouldBe result(1)
    result(2) shouldBe result(3)
    result should contain theSameElementsAs expectedEmailMap

    val vectorizedWhitelist = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, whiteListKeys = Array(emailKey), trackNulls = false)
    val resultWhitelist = transformAndCollect(ds1, vectorizedWhitelist)
    resultWhitelist(0) shouldBe resultWhitelist(1)
    resultWhitelist(2) shouldBe resultWhitelist(3)
    resultWhitelist should contain theSameElementsAs expectedEmailMap
  }

  it should "track nulls with whitelisted/ignore blacklisted keys in EmailMap" in {


    val (ds1, f1) = TestFeatureBuilder(emailMap)
    val vectorized = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, blackListKeys = Array(emailKey2), trackNulls = true)

    val result = transformAndCollect(ds1, vectorized)
    result(0) shouldBe result(1)
    result(2) shouldBe result(3)
    result should contain theSameElementsAs expectedTrackNulls

    val vectorizedWhitelist = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, whiteListKeys = Array(emailKey), trackNulls = true)
    val resultWhitelist = transformAndCollect(ds1, vectorizedWhitelist)
    resultWhitelist(0) shouldBe resultWhitelist(1)
    resultWhitelist(2) shouldBe resultWhitelist(3)
    resultWhitelist should contain theSameElementsAs expectedTrackNulls
  }

  Spec[RichEmailFeature] should "vectorize Emails correctly" in {
    val (ds2, f2) = TestFeatureBuilder(emails)
    val vectorized = f2.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText)
    vectorized.originStage shouldBe a[OpTextPivotVectorizer[_]]
    vectorized shouldBe a[FeatureLike[_]]

    val result = transformAndCollect(ds2, vectorized)
    result(0) shouldBe result(1)
    result(2) shouldBe result(3)
    result should contain theSameElementsAs expectedEmail
  }
}
