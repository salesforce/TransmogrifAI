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

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.TestOpVectorColumnType.IndColWithGroup
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder, TestOpVectorMetadataBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichStructType._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class RealMapVectorizerTest
  extends OpEstimatorSpec[OPVector, SequenceModel[RealMap, OPVector], RealMapVectorizer[RealMap]]
    with AttributeAsserts {

  val (inputData, m1, m2) = TestFeatureBuilder("m1", "m2",
    Seq(
      (Map("a" -> 1.0, "b" -> 5.0), Map("z" -> 10.0)),
      (Map("c" -> 11.0), Map("y" -> 3.0, "x" -> 0.0)),
      (Map.empty[String, Double], Map.empty[String, Double])
    ).map(v => v._1.toRealMap -> v._2.toRealMap)
  )

  val (meanData, f1, f2) = TestFeatureBuilder("f1", "f2",
    Seq(
      (Map("a" -> 1.0, "b" -> 5.0), Map("y" -> 4.0, "x" -> 0.0, "z" -> 10.0)),
      (Map("a" -> -3.0, "b" -> 3.0, "c" -> 11.0), Map("y" -> 3.0, "x" -> 0.0)),
      (Map.empty[String, Double], Map("y" -> 1.0, "x" -> 0.0, "z" -> 5.0)),
      (Map.empty[String, Double], Map.empty[String, Double])
    ).map(v => v._1.toRealMap -> v._2.toRealMap)
  )

  val estimator = new RealMapVectorizer[RealMap]().setInput(m1, m2).setTrackNulls(false).setCleanKeys(true)

  val expectedResult = Seq(
    Vectors.dense(Array(1.0, 5.0, 0.0, 0.0, 0.0, 10.0)),
    Vectors.sparse(6, Array(2, 4), Array(11.0, 3.0)),
    Vectors.sparse(6, Array(), Array())
  ).map(_.toOPVector)

  val expectedMeta = TestOpVectorMetadataBuilder(
    estimator,
    m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B"), IndColWithGroup(None, "C")),
    m2 -> List(IndColWithGroup(None, "X"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "Z"))
  )

  val nullIndicatorValue = Some(OpVectorColumnMetadata.NullString)
  val expectedMetaTrackNulls = TestOpVectorMetadataBuilder(
    estimator,
    m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(nullIndicatorValue, "A"),
      IndColWithGroup(None, "B"), IndColWithGroup(nullIndicatorValue, "B"),
      IndColWithGroup(None, "C"), IndColWithGroup(nullIndicatorValue, "C")),
    m2 -> List(IndColWithGroup(None, "X"), IndColWithGroup(nullIndicatorValue, "X"),
      IndColWithGroup(None, "Y"), IndColWithGroup(nullIndicatorValue, "Y"),
      IndColWithGroup(None, "Z"), IndColWithGroup(nullIndicatorValue, "Z"))
  )

  it should "return a model that correctly transforms the data and produces metadata" in {
    val vectorizer = estimator.setDefaultValue(0.0).setTrackNulls(false).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expectedResult.head.value.size)(false))
    transformed.collect(vector) shouldBe expectedResult
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls" in {
    val vectorizer = estimator.setDefaultValue(0.0).setTrackNulls(true).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(12, Array(0, 2, 5, 7, 9, 10), Array(1.0, 5.0, 1.0, 1.0, 1.0, 10.0)),
      Vectors.sparse(12, Array(1, 3, 4, 8, 11), Array(1.0, 1.0, 11.0, 3.0, 1.0)),
      Vectors.sparse(12, Array(1, 3, 5, 7, 9, 11), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size / 2)(Seq(false, true)).flatten)
    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMetaTrackNulls
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMetaTrackNulls
  }

  it should "use the correct fill value for missing keys" in {
    val vectorizer = estimator.setDefaultValue(100).setTrackNulls(false).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(Array(1.0, 5.0, 100.0, 100.0, 100.0, 10.0)),
      Vectors.dense(Array(100.0, 100.0, 11.0, 0.0, 3.0, 100.0)),
      Vectors.dense(Array(100.0, 100.0, 100.0, 100.0, 100.0, 100.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expectedResult.head.value.size)(false))
    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with fill value for missing keys" in {
    val vectorizer = estimator.setDefaultValue(100).setTrackNulls(true).fit(inputData)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(inputData)

    val expected = Array(
      Vectors.dense(Array(1.0, 0.0, 5.0, 0.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 10.0, 0.0)),
      Vectors.dense(Array(100.0, 1.0, 100.0, 1.0, 11.0, 0.0, 0.0, 0.0, 3.0, 0.0, 100.0, 1.0)),
      Vectors.dense(Array(100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size / 2)(Seq(false, true)).flatten)
    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMetaTrackNulls
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMetaTrackNulls
  }

  it should "correctly whitelist keys" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2).setDefaultValue(0.0)
      .setCleanKeys(true).setWhiteListKeys(Array("a", "b", "z")).setTrackNulls(false).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(Array(1.0, 5.0, 10.0)),
      Vectors.sparse(3, Array(), Array()),
      Vectors.sparse(3, Array(), Array())
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B")),
      m2 -> List(IndColWithGroup(None, "Z")))
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size)(false))
    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with whitelist keys" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2).setDefaultValue(0.0)
      .setCleanKeys(true).setWhiteListKeys(Array("a", "b", "z")).setTrackNulls(true).fit(inputData)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(inputData)

    val expected = Array(
      Vectors.sparse(6, Array(0, 2, 4), Array(1.0, 5.0, 10.0)),
      Vectors.sparse(6, Array(1, 3, 5), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(6, Array(1, 3, 5), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(nullIndicatorValue, "A"),
        IndColWithGroup(None, "B"), IndColWithGroup(nullIndicatorValue, "B")),
      m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(nullIndicatorValue, "Z"))
    )
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size / 2)(Seq(false, true)).flatten)
    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly backlist keys" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2).setDefaultValue(0.0)
      .setCleanKeys(true).setBlackListKeys(Array("a", "z")).setTrackNulls(false).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(4, Array(0), Array(5.0)),
      Vectors.dense(Array(0.0, 11.0, 0.0, 3.0)),
      Vectors.sparse(4, Array(), Array())
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "B"), IndColWithGroup(None, "C")),
      m2 -> List(IndColWithGroup(None, "X"), IndColWithGroup(None, "Y"))
    )
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size)(false))
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with backlist keys" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2).setDefaultValue(0.0)
      .setCleanKeys(true).setBlackListKeys(Array("a", "z")).setTrackNulls(true).fit(inputData)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(inputData)

    val expected = Array(
      Vectors.sparse(8, Array(0, 3, 5, 7), Array(5.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(8, Array(1, 2, 6), Array(1.0, 11.0, 3.0)),
      Vectors.sparse(8, Array(1, 3, 5, 7), Array(1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "B"), IndColWithGroup(nullIndicatorValue, "B"),
        IndColWithGroup(None, "C"), IndColWithGroup(nullIndicatorValue, "C")),
      m2 -> List(IndColWithGroup(None, "X"), IndColWithGroup(nullIndicatorValue, "X"),
        IndColWithGroup(None, "Y"), IndColWithGroup(nullIndicatorValue, "Y"))
    )
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size / 2)(Seq(false, true)).flatten)
    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly calculate means by key and fill missing values with them" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(f1, f2).setCleanKeys(true)
      .setFillWithMean(true).setTrackNulls(false).fit(meanData)
    val transformed = vectorizer.transform(meanData)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(1.0, 5.0, 11.0, 0.0, 4.0, 10.0),
      Vectors.dense(-3.0, 3.0, 11.0, 0.0, 3.0, 15.0 / 2),
      Vectors.dense(-1.0, 4.0, 11.0, 0.0, 1.0, 5.0),
      Vectors.dense(-1.0, 4.0, 11.0, 0.0, 8.0 / 3, 15.0 / 2)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expectedResult.head.value.size)(false))
    transformed.collect(vector) shouldBe expected

    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B"), IndColWithGroup(None, "C")),
      f2 -> List(IndColWithGroup(None, "X"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "Z"))
    )
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with means by key" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(f1, f2).setCleanKeys(true)
      .setFillWithMean(true).setTrackNulls(true).fit(meanData)
    val transformed = vectorizer.transform(meanData)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(12, Array(0, 2, 4, 5, 8, 10), Array(1.0, 5.0, 11.0, 1.0, 4.0, 10.0)),
      Vectors.sparse(12, Array(0, 2, 4, 8, 10, 11), Array(-3.0, 3.0, 11.0, 3.0, 15.0 / 2, 1.0)),
      Vectors.dense(-1.0, 1.0, 4.0, 1.0, 11.0, 1.0, 0.0, 0.0, 1.0, 0.0, 5.0, 0.0),
      Vectors.dense(-1.0, 1.0, 4.0, 1.0, 11.0, 1.0, 0.0, 1.0, 8.0 / 3, 1.0, 15.0 / 2, 1.0)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size / 2)(Seq(false, true)).flatten)
    transformed.collect(vector) shouldBe expected

    val expectedMetaTrackNulls = TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(nullIndicatorValue, "A"),
        IndColWithGroup(None, "B"), IndColWithGroup(nullIndicatorValue, "B"),
        IndColWithGroup(None, "C"), IndColWithGroup(nullIndicatorValue, "C")),
      f2 -> List(IndColWithGroup(None, "X"), IndColWithGroup(nullIndicatorValue, "X"),
        IndColWithGroup(None, "Y"), IndColWithGroup(nullIndicatorValue, "Y"),
        IndColWithGroup(None, "Z"), IndColWithGroup(nullIndicatorValue, "Z"))
    )
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMetaTrackNulls
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMetaTrackNulls
  }

}
