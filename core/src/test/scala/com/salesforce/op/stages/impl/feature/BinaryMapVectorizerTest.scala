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

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, IndColWithGroup, IndVal}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class BinaryMapVectorizerTest extends FlatSpec with TestSparkContext {

  lazy val (data, m1, m2) = TestFeatureBuilder("m1", "m2",
    Seq(
      (Map("a" -> false, "b" -> true), Map("z" -> false)),
      (Map("c" -> false), Map("y" -> true, "x" -> true)),
      (Map.empty[String, Boolean], Map.empty[String, Boolean])
    ).map(v => v._1.toBinaryMap -> v._2.toBinaryMap)
  )
  val vectorizer = new BinaryMapVectorizer().setInput(m1, m2).setCleanKeys(true)

  /**
   * Note that defaults and filters are tested in [[RealMapVectorizerTest]]
   * as that code is shared between the two classes
   */
  Spec[BinaryMapVectorizer[_]] should "take an array of features as input and return a single vector feature" in {
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.getOutputFeatureName
    vector.parents should contain theSameElementsAs Array(m1, m2)
    vector.originStage shouldBe vectorizer
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return a model that correctly transforms the data" in {
    val transformed = vectorizer.setTrackNulls(false).fit(data).transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(6, Array(1), Array(1.0)),
      Vectors.sparse(6, Array(4, 5), Array(1.0, 1.0)),
      Vectors.sparse(6, Array(), Array())
    ).map(_.toOPVector)

    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B"),
        IndColWithGroup(None, "C")),
      m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "X"))
    )

    transformed.collect(vector) shouldBe expected
    val field = transformed.schema(vectorizer.getOutputFeatureName)
    OpVectorMetadata(field) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(field.copy(metadata = vectorMetadata)) shouldEqual expectedMeta
  }

  it should " track nulls" in {
    val transformed = vectorizer.setTrackNulls(true).fit(data).transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(12, Array(2, 5, 9, 11), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(1, 3, 7, 8, 10), Array(1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(1, 3, 5, 7, 9, 11), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)


    val nullIndicatorValue = Some(OpVectorColumnMetadata.NullString)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(nullIndicatorValue, "A"),
        IndColWithGroup(None, "B"), IndColWithGroup(nullIndicatorValue, "B"),
        IndColWithGroup(None, "C"), IndColWithGroup(nullIndicatorValue, "C")),
      m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(nullIndicatorValue, "Z"),
        IndColWithGroup(None, "Y"), IndColWithGroup(nullIndicatorValue, "Y"),
        IndColWithGroup(None, "X"), IndColWithGroup(nullIndicatorValue, "X"))
    )

    transformed.collect(vector) shouldBe expected
    val field = transformed.schema(vectorizer.getOutputFeatureName)
    OpVectorMetadata(field) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(field.copy(metadata = vectorMetadata)) shouldEqual expectedMeta
  }

}
