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

import com.salesforce.op.features.types.{TextMap, _}
import com.salesforce.op.test.TestOpVectorColumnType.DescColWithGroup
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TextMapLenEstimatorTest extends FlatSpec with TestSparkContext with AttributeAsserts {

  val (ds, f1) = TestFeatureBuilder(
    Seq[TextMap](
      TextMap(Map("k1" -> "A giraffe drinks by the watering hole", "k2" -> "Cheese", "k3" -> "Hello", "k4" -> "Bye")),
      // scalastyle:off
      TextMap(Map("k2" -> "French Fries", "k4" -> "\uA7BC\u10C8\u2829\u29BA\u23E1")),
      // scalastyle:on
      TextMap(Map("k3" -> "Hip-hop Pottamus"))
    )
  )

  Spec[TextMapLenEstimator[_]] should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new TextMapLenEstimator[TextMap]().setInput(f1)
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.getOutputFeatureName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "transform the data correctly" in {
    val vectorizer = new TextMapLenEstimator[TextMap]().setInput(f1)
    val transformed = vectorizer.fit(ds).transform(ds)
    val vector = vectorizer.getOutput()

    val expected = Array(
      Array(31.0, 6.0, 5.0, 3.0),
      Array(0.0, 11.0, 0.0, 0.0),
      Array(0.0, 0.0, 14.0, 0.0)
    ).map(Vectors.dense(_).toOPVector)
    val result = transformed.collect(vector)
    result shouldBe expected
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size)(false), result)
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(
        DescColWithGroup(name = Option(TransmogrifierDefaults.TextLenString), groupName = "k1"),
        DescColWithGroup(name = Option(TransmogrifierDefaults.TextLenString), groupName = "k2"),
        DescColWithGroup(name = Option(TransmogrifierDefaults.TextLenString), groupName = "k3"),
        DescColWithGroup(name = Option(TransmogrifierDefaults.TextLenString), groupName = "k4")
      )
    )
  }

  it should "transform the data correctly when the text is not tokenized" in {
    val vectorizer = new TextMapLenEstimator[TextMap]().setInput(f1).setTokenizeText(false)
    val transformed = vectorizer.fit(ds).transform(ds)
    val vector = vectorizer.getOutput()

    val expected = Array(
      Array(31.0, 6.0, 5.0, 3.0),
      Array(0.0, 11.0, 0.0, 5.0),
      Array(0.0, 0.0, 14.0, 0.0)
    ).map(Vectors.dense(_).toOPVector)
    val result = transformed.collect(vector)
    result shouldBe expected
  }

  it should "transform the data correctly when the text values is not cleaned" in {
    val vectorizer = new TextMapLenEstimator[TextMap]().setInput(f1).setCleanText(false).setTokenizeText(false)
    val transformed = vectorizer.fit(ds).transform(ds)
    val vector = vectorizer.getOutput()

    val expected = Array(
      Array(37.0, 6.0, 5.0, 3.0),
      Array(0.0, 12.0, 0.0, 5.0),
      Array(0.0, 0.0, 16.0, 0.0)
    ).map(Vectors.dense(_).toOPVector)
    val result = transformed.collect(vector)
    result shouldBe expected
  }
}
