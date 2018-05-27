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
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, IndColWithGroup}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichStructType._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FlatSpec


@RunWith(classOf[JUnitRunner])
class GeolocationMapVectorizerTest extends FlatSpec with TestSparkContext {

  lazy val (data, m1, m2) = TestFeatureBuilder("m1", "m2",
    Seq(
      (Map("a" -> Seq(32.4, -100.2, 3.0), "b" -> Seq(33.8, -108.7, 2.0)), Map("z" -> Seq(45.0, -105.5, 4.0))),
      (Map("c" -> Seq(33.8, -108.7, 2.0)), Map("y" -> Seq(42.5, -95.4, 4.0), "x" -> Seq(40.4, -116.3, 2.0))),
      (Map.empty[String, Seq[Double]], Map.empty[String, Seq[Double]])
    ).map(v => (GeolocationMap(v._1), GeolocationMap(v._2)))
  )

  val baseVectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true)
  val expectedMeta = TestOpVectorMetadataBuilder(
    baseVectorizer,
    m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "A"), IndColWithGroup(None, "A"),
      IndColWithGroup(None, "B"), IndColWithGroup(None, "B"), IndColWithGroup(None, "B"),
      IndColWithGroup(None, "C"), IndColWithGroup(None, "C"), IndColWithGroup(None, "C")),
    m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(None, "Z"), IndColWithGroup(None, "Z"),
      IndColWithGroup(None, "Y"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "Y"),
      IndColWithGroup(None, "X"), IndColWithGroup(None, "X"), IndColWithGroup(None, "X"))
  )
  val nullIndicatorValue = Some(OpVectorColumnMetadata.NullString)

  val expectedMetaTrackNulls = TestOpVectorMetadataBuilder(
    baseVectorizer,
    m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "A"), IndColWithGroup(None, "A"),
      IndColWithGroup(nullIndicatorValue, "A"),
      IndColWithGroup(None, "B"), IndColWithGroup(None, "B"), IndColWithGroup(None, "B"),
      IndColWithGroup(nullIndicatorValue, "B"),
      IndColWithGroup(None, "C"), IndColWithGroup(None, "C"), IndColWithGroup(None, "C"),
      IndColWithGroup(nullIndicatorValue, "C")),
    m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(None, "Z"), IndColWithGroup(None, "Z"),
      IndColWithGroup(nullIndicatorValue, "Z"),
      IndColWithGroup(None, "Y"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "Y"),
      IndColWithGroup(nullIndicatorValue, "Y"),
      IndColWithGroup(None, "X"), IndColWithGroup(None, "X"), IndColWithGroup(None, "X"),
      IndColWithGroup(nullIndicatorValue, "X"))
  )

  Spec[GeolocationMapVectorizer] should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true)
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.getOutputFeatureName
    vector.parents should contain theSameElementsAs Array(m1, m2)
    vector.originStage shouldBe vectorizer
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return a model that correctly transforms the data" in {
    val vectorizer = baseVectorizer.setTrackNulls(false).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(18, Array(0, 1, 2, 3, 4, 5, 9, 10, 11),
        Array(32.4, -100.2, 3.0, 33.8, -108.7, 2.0, 45.0, -105.5, 4.0)),
      Vectors.sparse(18, Array(6, 7, 8, 12, 13, 14, 15, 16, 17),
        Array(33.8, -108.7, 2.0, 42.5, -95.4, 4.0, 40.4, -116.3, 2.0)),
      Vectors.sparse(18, Array(), Array())
    ).map(_.toOPVector)


    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls" in {
    val vectorizer = baseVectorizer.setTrackNulls(true).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(24, Array(0, 1, 2, 4, 5, 6, 11, 12, 13, 14, 19, 23),
        Array(32.4, -100.2, 3.0, 33.8, -108.7, 2.0, 1.0, 45.0, -105.5, 4.0, 1.0, 1.0)),
      Vectors.sparse(24, Array(3, 7, 8, 9, 10, 15, 16, 17, 18, 20, 21, 22),
        Array(1.0, 1.0, 33.8, -108.7, 2.0, 1.0, 42.5, -95.4, 4.0, 40.4, -116.3, 2.0)),
      Vectors.sparse(24, Array(3, 7, 11, 15, 19, 23), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)


    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMetaTrackNulls
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMetaTrackNulls
  }

  it should "use the correct fill value for missing keys" in {
    val vectorizer = baseVectorizer.setTrackNulls(false)
      .setDefaultValue(Geolocation(6.0, 6.0, GeolocationAccuracy.Zip)).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(Array(32.4, -100.2, 3.0, 33.8, -108.7, 2.0, 6.0, 6.0, 6.0, 45.0, -105.5, 4.0, 6.0, 6.0, 6.0,
        6.0, 6.0, 6.0)),
      Vectors.dense(Array(6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 33.8, -108.7, 2.0, 6.0, 6.0, 6.0, 42.5, -95.4, 4.0,
        40.4, -116.3, 2.0)),
      Vectors.dense(Array.fill(18)(6.0))
    ).map(_.toOPVector)

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with missing keys" in {
    val vectorizer = baseVectorizer.setTrackNulls(true)
      .setDefaultValue(Geolocation(6.0, 6.0, GeolocationAccuracy.Zip)).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(Array(32.4, -100.2, 3.0, 0.0, 33.8, -108.7, 2.0, 0.0, 6.0, 6.0, 6.0, 1.0, 45.0, -105.5, 4.0, 0.0,
        6.0, 6.0, 6.0, 1.0, 6.0, 6.0, 6.0, 1.0)),
      Vectors.dense(Array(6.0, 6.0, 6.0, 1.0, 6.0, 6.0, 6.0, 1.0, 33.8, -108.7, 2.0, 0.0, 6.0, 6.0, 6.0, 1.0,
        42.5, -95.4, 4.0, 0.0, 40.4, -116.3, 2.0, 0.0)),
      Vectors.dense((0 until 6).flatMap(k => Seq.fill(3)(6.0) :+ 1.0).toArray)
    ).map(_.toOPVector)

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMetaTrackNulls
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMetaTrackNulls
  }

  it should "correctly whitelist keys" in {
    val vectorizer = new GeolocationMapVectorizer().setCleanKeys(true).setTrackNulls(false)
      .setInput(m1, m2).setWhiteListKeys(Array("a", "b", "z")).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(Array(32.4, -100.2, 3.0, 33.8, -108.7, 2.0, 45.0, -105.5, 4.0)),
      Vectors.sparse(9, Array(), Array()),
      Vectors.sparse(9, Array(), Array())
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "A"), IndColWithGroup(None, "A"),
        IndColWithGroup(None, "B"), IndColWithGroup(None, "B"), IndColWithGroup(None, "B")),
      m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(None, "Z"), IndColWithGroup(None, "Z"))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with whitelist keys" in {
    val vectorizer = new GeolocationMapVectorizer().setCleanKeys(true).setTrackNulls(true)
      .setInput(m1, m2).setWhiteListKeys(Array("a", "b", "z")).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(Array(32.4, -100.2, 3.0, 0.0, 33.8, -108.7, 2.0, 0.0, 45.0, -105.5, 4.0, 0.0)),
      Vectors.sparse(12, Array(3, 7, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(3, 7, 11), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "A"), IndColWithGroup(None, "A"),
        IndColWithGroup(nullIndicatorValue, "A"),
        IndColWithGroup(None, "B"), IndColWithGroup(None, "B"), IndColWithGroup(None, "B"),
        IndColWithGroup(nullIndicatorValue, "B")),
      m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(None, "Z"), IndColWithGroup(None, "Z"),
        IndColWithGroup(nullIndicatorValue, "Z"))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly backlist keys" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true).setTrackNulls(false)
      .setBlackListKeys(Array("a", "z")).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(12, Array(0, 1, 2), Array(33.8, -108.7, 2.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 33.8, -108.7, 2.0, 42.5, -95.4, 4.0, 40.4, -116.3, 2.0)),
      Vectors.sparse(12, Array(), Array())
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "B"), IndColWithGroup(None, "B"), IndColWithGroup(None, "B"),
        IndColWithGroup(None, "C"), IndColWithGroup(None, "C"), IndColWithGroup(None, "C")),
      m2 -> List(IndColWithGroup(None, "Y"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "Y"),
        IndColWithGroup(None, "X"), IndColWithGroup(None, "X"), IndColWithGroup(None, "X"))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with backlist keys" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true).setTrackNulls(true)
      .setBlackListKeys(Array("a", "z")).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(16, Array(0, 1, 2, 7, 11, 15), Array(33.8, -108.7, 2.0, 1.0, 1.0, 1.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 1.0, 33.8, -108.7, 2.0, 0.0, 42.5, -95.4, 4.0, 0.0, 40.4, -116.3, 2.0, 0.0)),
      Vectors.sparse(16, Array(3, 7, 11, 15), Array(1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "B"), IndColWithGroup(None, "B"), IndColWithGroup(None, "B"),
        IndColWithGroup(nullIndicatorValue, "B"),
        IndColWithGroup(None, "C"), IndColWithGroup(None, "C"), IndColWithGroup(None, "C"),
        IndColWithGroup(nullIndicatorValue, "C")),
      m2 -> List(IndColWithGroup(None, "Y"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "Y"),
        IndColWithGroup(nullIndicatorValue, "Y"),
        IndColWithGroup(None, "X"), IndColWithGroup(None, "X"), IndColWithGroup(None, "X"),
        IndColWithGroup(nullIndicatorValue, "X"))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "have a working shortcut function" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expectedOutput = transformed.collect()

    // Now using the shortcut
    val res = m1.vectorize(cleanKeys = TransmogrifierDefaults.CleanKeys, others = Array(m2))
    val actualOutput = res.originStage.asInstanceOf[GeolocationMapVectorizer]
      .fit(data).transform(data).collect()

    actualOutput.zip(expectedOutput).forall(f => f._1 == f._2) shouldBe true
  }
}
