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
import com.salesforce.op.test.TestOpVectorColumnType.{DescColWithGroup, IndColWithGroup}
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder, TestOpVectorMetadataBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichStructType._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class GeolocationMapVectorizerTest
  extends OpEstimatorSpec[OPVector, SequenceModel[GeolocationMap, OPVector], GeolocationMapVectorizer]
    with AttributeAsserts {

  val (inputData, m1, m2) = TestFeatureBuilder("m1", "m2",
    Seq(
      (Map("a" -> Seq(32.4, -100.2, 3.0), "b" -> Seq(33.8, -108.7, 2.0)), Map("z" -> Seq(45.0, -105.5, 4.0))),
      (Map("c" -> Seq(33.8, -108.7, 2.0)), Map("y" -> Seq(42.5, -95.4, 4.0), "x" -> Seq(40.4, -116.3, 2.0))),
      (Map.empty[String, Seq[Double]], Map.empty[String, Seq[Double]])
    ).map(v => (GeolocationMap(v._1), GeolocationMap(v._2)))
  )

  val estimator = new GeolocationMapVectorizer().setInput(m1, m2).setTrackNulls(false).setCleanKeys(true)

  val expectedResult = Seq(
    Vectors.sparse(18, Array(0, 1, 2, 3, 4, 5, 15, 16, 17),
      Array(32.4, -100.2, 3.0, 33.8, -108.7, 2.0, 45.0, -105.5, 4.0)),
    Vectors.sparse(18, Array(6, 7, 8, 9, 10, 11, 12, 13, 14),
      Array(33.8, -108.7, 2.0, 40.4, -116.3, 2.0, 42.5, -95.4, 4.0)),
    Vectors.sparse(18, Array(), Array())
  ).map(_.toOPVector)

  val expectedMeta = TestOpVectorMetadataBuilder(
    estimator,
    m1 -> (Geolocation.Names.map(n => DescColWithGroup(Option(n), "A")) ++
      Geolocation.Names.map(n => DescColWithGroup(Option(n), "B")) ++
      Geolocation.Names.map(n => DescColWithGroup(Option(n), "C"))).toList,
    m2 -> (Geolocation.Names.map(n => DescColWithGroup(Option(n), "X")) ++
      Geolocation.Names.map(n => DescColWithGroup(Option(n), "Y")) ++
      Geolocation.Names.map(n => DescColWithGroup(Option(n), "Z"))).toList
  )
  val nullIndicatorValue = Some(OpVectorColumnMetadata.NullString)

  val expectedMetaTrackNulls = TestOpVectorMetadataBuilder(
    estimator,
    m1 -> (
      (Geolocation.Names.map(n => DescColWithGroup(Option(n), "A")) :+ IndColWithGroup(nullIndicatorValue, "A")) ++
        (Geolocation.Names.map(n => DescColWithGroup(Option(n), "B")) :+ IndColWithGroup(nullIndicatorValue, "B")) ++
        Geolocation.Names.map(n => DescColWithGroup(Option(n), "C")) :+ IndColWithGroup(nullIndicatorValue, "C")
      ).toList,
    m2 -> (
      (Geolocation.Names.map(n => DescColWithGroup(Option(n), "X")) :+ IndColWithGroup(nullIndicatorValue, "X")) ++
        (Geolocation.Names.map(n => DescColWithGroup(Option(n), "Y")) :+ IndColWithGroup(nullIndicatorValue, "Y")) ++
        Geolocation.Names.map(n => DescColWithGroup(Option(n), "Z")) :+ IndColWithGroup(nullIndicatorValue, "Z")
      ).toList
  )

  it should "return a model that correctly transforms the data" in {
    val vectorizer = estimator.fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expectedResult.head.value.size)(false), result)
    result shouldBe expectedResult
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls" in {
    val vectorizer = estimator.setTrackNulls(true).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.sparse(24, Array(0, 1, 2, 4, 5, 6, 11, 15, 19, 20, 21, 22),
        Array(32.4, -100.2, 3.0, 33.8, -108.7, 2.0, 1.0, 1.0, 1.0, 45.0, -105.5, 4.0)),
      Vectors.sparse(24, Array(3, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 23),
        Array(1.0, 1.0, 33.8, -108.7, 2.0, 40.4, -116.3, 2.0, 42.5, -95.4, 4.0, 1.0)),
      Vectors.sparse(24, Array(3, 7, 11, 15, 19, 23), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)

    val field = transformed.schema(vector.name)
    assertNominal(field,
      Array.fill(expected.head.value.size / 4)(Seq(false, false, false, true)).flatten, result)
    result shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMetaTrackNulls
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMetaTrackNulls
  }

  it should "use the correct fill value for missing keys" in {
    val vectorizer = estimator.setTrackNulls(false)
      .setDefaultValue(Geolocation(6.0, 6.0, GeolocationAccuracy.Zip)).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val expected = Array(
      Array(32.4, -100.2, 3.0, 33.8, -108.7, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 45.0, -105.5, 4.0),
      Array(6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 33.8, -108.7, 2.0, 40.4, -116.3, 2.0, 42.5, -95.4, 4.0, 6.0, 6.0, 6.0),
      Array.fill(18)(6.0)
    ).map(v => Vectors.dense(v).toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size)(false), result)

    result shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with missing keys" in {
    val vectorizer = estimator.setTrackNulls(true)
      .setDefaultValue(Geolocation(6.0, 6.0, GeolocationAccuracy.Zip)).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val expected = Array(
      Array(32.4, -100.2, 3.0, 0.0, 33.8, -108.7, 2.0, 0.0, 6.0, 6.0, 6.0, 1.0, 6.0, 6.0, 6.0, 1.0, 6.0, 6.0, 6.0, 1.0,
        45.0, -105.5, 4.0, 0.0),
      Array(6.0, 6.0, 6.0, 1.0, 6.0, 6.0, 6.0, 1.0, 33.8, -108.7, 2.0, 0.0, 40.4, -116.3, 2.0, 0.0, 42.5, -95.4, 4.0,
        0.0, 6.0, 6.0, 6.0, 1.0),
      (0 until 6).flatMap(k => Seq.fill(3)(6.0) :+ 1.0).toArray
    ).map(v => Vectors.dense(v).toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field,
      Array.fill(expected.head.value.size / 4)(Seq(false, false, false, true)).flatten, result)

    result shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMetaTrackNulls
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMetaTrackNulls
  }

  it should "correctly allowlist keys" in {
    val vectorizer = new GeolocationMapVectorizer().setCleanKeys(true).setTrackNulls(false)
      .setInput(m1, m2).setAllowListKeys(Array("a", "b", "z")).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.dense(Array(32.4, -100.2, 3.0, 33.8, -108.7, 2.0, 45.0, -105.5, 4.0)),
      Vectors.sparse(9, Array(), Array()),
      Vectors.sparse(9, Array(), Array())
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size)(false), result)

    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> (Geolocation.Names.map(n => DescColWithGroup(Option(n), "A")) ++
        Geolocation.Names.map(n => DescColWithGroup(Option(n), "B"))
        ).toList,
      m2 -> Geolocation.Names.map(n => DescColWithGroup(Option(n), "Z")).toList
    )

    result shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with allowlist keys" in {
    val vectorizer = new GeolocationMapVectorizer().setCleanKeys(true).setTrackNulls(true)
      .setInput(m1, m2).setAllowListKeys(Array("a", "b", "z")).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.dense(Array(32.4, -100.2, 3.0, 0.0, 33.8, -108.7, 2.0, 0.0, 45.0, -105.5, 4.0, 0.0)),
      Vectors.sparse(12, Array(3, 7, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(3, 7, 11), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field,
      Array.fill(expected.head.value.size / 4)(Seq(false, false, false, true)).flatten, result)

    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> (
        (Geolocation.Names.map(n => DescColWithGroup(Option(n), "A")) :+ IndColWithGroup(nullIndicatorValue, "A")) ++
          Geolocation.Names.map(n => DescColWithGroup(Option(n), "B")) :+ IndColWithGroup(nullIndicatorValue, "B")
        ).toList,
      m2 -> (
        Geolocation.Names.map(n => DescColWithGroup(Option(n), "Z")) :+ IndColWithGroup(nullIndicatorValue, "Z")
        ).toList
    )

    result shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly backlist keys" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true).setTrackNulls(false)
      .setDenyListKeys(Array("a", "z")).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.sparse(12, Array(0, 1, 2), Array(33.8, -108.7, 2.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 33.8, -108.7, 2.0, 40.4, -116.3, 2.0, 42.5, -95.4, 4.0)),
      Vectors.sparse(12, Array(), Array())
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected.head.value.size)(false), result)

    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> (Geolocation.Names.map(n => DescColWithGroup(Option(n), "B")) ++
        Geolocation.Names.map(n => DescColWithGroup(Option(n), "C"))).toList,
      m2 -> (Geolocation.Names.map(n => DescColWithGroup(Option(n), "X")) ++
        Geolocation.Names.map(n => DescColWithGroup(Option(n), "Y"))).toList
    )

    result shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with backlist keys" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true).setTrackNulls(true)
      .setDenyListKeys(Array("a", "z")).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.sparse(16, Array(0, 1, 2, 7, 11, 15), Array(33.8, -108.7, 2.0, 1.0, 1.0, 1.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 1.0, 33.8, -108.7, 2.0, 0.0, 40.4, -116.3, 2.0, 0.0, 42.5, -95.4, 4.0, 0.0)),
      Vectors.sparse(16, Array(3, 7, 11, 15), Array(1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    assertNominal(field,
      Array.fill(expected.head.value.size / 4)(Seq(false, false, false, true)).flatten, result)

    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> (
        (Geolocation.Names.map(n => DescColWithGroup(Option(n), "B")) :+ IndColWithGroup(nullIndicatorValue, "B")) ++
          Geolocation.Names.map(n => DescColWithGroup(Option(n), "C")) :+ IndColWithGroup(nullIndicatorValue, "C")
        ).toList,
      m2 -> (
        (Geolocation.Names.map(n => DescColWithGroup(Option(n), "X")) :+ IndColWithGroup(nullIndicatorValue, "X")) ++
          Geolocation.Names.map(n => DescColWithGroup(Option(n), "Y")) :+ IndColWithGroup(nullIndicatorValue, "Y")
        ).toList
    )

    result shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "have a working shortcut function" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true).fit(inputData)
    val transformed = vectorizer.transform(inputData)
    val vector = vectorizer.getOutput()
    val expectedOutput = transformed.collect()
    val result = transformed.collect(vector)

    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(result.head.value.size / 4)
    (Seq(false, false, false, true)).flatten, result)
    // Now using the shortcut
    val res = m1.vectorize(cleanKeys = TransmogrifierDefaults.CleanKeys, others = Array(m2))
    res.originStage shouldBe a[GeolocationMapVectorizer]
    val actualOutput = res.originStage.asInstanceOf[GeolocationMapVectorizer].fit(inputData)
      .transform(inputData).collect()

    actualOutput.zip(expectedOutput).forall(f => f._1 == f._2) shouldBe true
  }
}
