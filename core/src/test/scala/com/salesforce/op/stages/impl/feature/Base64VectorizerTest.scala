/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class Base64VectorizerTest extends FlatSpec with TestSparkContext with Base64TestData {

  "Base64Vectorizer" should "vectorize random binary data" in {
    val vec = randomBase64.vectorize(topK = 10, minSupport = 0, cleanText = true, trackNulls = false)
    val result = new OpWorkflow().setResultFeatures(vec).transform(randomData)

    result.collect(vec) should contain theSameElementsInOrderAs
      OPVector(Vectors.dense(0.0, 0.0)) +: Array.fill(expectedRandom.length - 1)(OPVector(Vectors.dense(1.0, 0.0)))
  }
  it should "vectorize some real binary content" in {
    val vec = realBase64.vectorize(topK = 10, minSupport = 0, cleanText = true)
    assertVectorizer(vec, expectedMime)
  }
  it should "vectorize some real binary content with a type hint" in {
    val vec = realBase64.vectorize(topK = 10, minSupport = 0, cleanText = true, typeHint = Some("application/json"))
    assertVectorizer(vec, expectedMimeJson)
  }

  def assertVectorizer(vec: FeatureLike[OPVector], expected: Seq[Text]): Unit = {
    val result = new OpWorkflow().setResultFeatures(vec).transform(realData)
    val vectors = result.collect(vec)

    vectors.length shouldBe expected.length
    // TODO add a more robust check
  }

}
