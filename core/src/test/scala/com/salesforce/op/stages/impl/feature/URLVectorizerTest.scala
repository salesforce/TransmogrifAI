/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
class URLVectorizerTest
  extends FlatSpec with FeatureTestBase with RichTextFeature with RichMapFeature with RichFeature {
  val urlKey = "Url1"
  val urlKey2 = "Url2"
  val urls = (RandomText.urlsOn(_ => "salesforce.com").take(2) ++ RandomText.urlsOn(_ => "data.com").take(2)).toSeq
  val urls2 = (RandomText.urlsOn(_ => "data.com").take(2) ++ RandomText.urlsOn(_ => "salesforce.com").take(2)).toSeq
  val urlMap =
    urls.zip(urls2).map { case (one, two) => Map(urlKey -> one.value.get, urlKey2 -> two.value.get).toURLMap }

  private val CleanText = true
  private val CleanKeys = true
  private val MinSupport = 0
  private val TopK = 10

  val expectedUrlMap = Array(
    Vectors.dense(1.0, 0.0, 0.0),
    Vectors.dense(1.0, 0.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0)
  ).map(_.toOPVector)

  val expectedUrl = Array(
    Vectors.dense(1.0, 0.0, 0.0, 0.0),
    Vectors.dense(1.0, 0.0, 0.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0, 0.0)
  ).map(_.toOPVector)

  def transformAndCollect(ds: DataFrame, feature: FeatureLike[OPVector]): Array[OPVector] =
    new OpWorkflow().setResultFeatures(feature).transform(ds).collect(feature)

  Spec[RichURLMapFeature] should "vectorize UrlMaps correctly" in {
    val (ds1, f1) = TestFeatureBuilder(urls.map(e => Map(urlKey -> e.value.get).toURLMap))
    val vectorized = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys)
    vectorized.originStage shouldBe a[TextMapPivotVectorizer[_]]
    vectorized shouldBe a[FeatureLike[_]]

    val result = transformAndCollect(ds1, vectorized)
    result(0) shouldBe result(1)
    result(2) shouldBe result(3)
    result should contain theSameElementsAs expectedUrlMap
  }

  it should "work on multiple keys in UrlMap" in {
    val expectedMulti = Array(
      Vectors.dense(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )

    val (ds1, f1) = TestFeatureBuilder(urlMap)
    val vectorized = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys)

    val result = transformAndCollect(ds1, vectorized).map(_.value.toDense)
    result shouldBe expectedMulti
  }

  it should "use whitelisted/ignore blacklisted keys in UrlMap" in {
    val (ds1, f1) = TestFeatureBuilder(urlMap)
    val vectorized = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, blackListKeys = Array(urlKey2))

    val result = transformAndCollect(ds1, vectorized)
    result(0) shouldBe result(1)
    result(2) shouldBe result(3)
    result should contain theSameElementsAs expectedUrlMap

    val vectorizedWhitelist = f1.vectorize(topK = TopK, minSupport = MinSupport,
      cleanText = CleanText, cleanKeys = CleanKeys, whiteListKeys = Array(urlKey))
    val resultWhitelist = transformAndCollect(ds1, vectorizedWhitelist)
    resultWhitelist(0) shouldBe resultWhitelist(1)
    resultWhitelist(2) shouldBe resultWhitelist(3)
    resultWhitelist should contain theSameElementsAs expectedUrlMap
  }

  Spec[RichURLFeature] should "vectorize Urls correctly" in {
    val (ds2, f2) = TestFeatureBuilder(urls)
    val vectorized = f2.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText)
    vectorized.originStage shouldBe a[OpTextPivotVectorizer[_]]
    vectorized shouldBe a[FeatureLike[_]]

    val result = transformAndCollect(ds2, vectorized)
    result(0) shouldBe result(1)
    result(2) shouldBe result(3)
    result should contain theSameElementsAs expectedUrl
  }
}
