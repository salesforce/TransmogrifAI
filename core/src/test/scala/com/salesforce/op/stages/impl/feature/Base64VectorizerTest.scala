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
