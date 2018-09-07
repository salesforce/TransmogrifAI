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

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType.IndCol
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.testkit.RandomReal
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.SparkException
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class NumericBucketizerTest extends FlatSpec with TestSparkContext with AttributeAsserts {

  trait GenericTest {
    val numbers = Seq(Some(10.0), None, Some(3.0), Some(5.0), Some(6.0), None, Some(1.0), Some(0.0))
    val reals = numbers.map(n => new Real(n))
    val integrals = numbers.map(n => new Integral(n.map(_.toLong)))

    val splits = Array(0.0, 1.0, 5.0, 10.0, Double.PositiveInfinity)
    val bucketLabels = Some(Array[String]("0-1", "1-5", "5-10", "10-Infinity"))
    val expectedAns = Array(
      Vectors.dense(Array(0.0, 0.0, 0.0, 1.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 0.0)),
      Vectors.dense(Array(0.0, 1.0, 0.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 1.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 1.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 0.0)),
      Vectors.dense(Array(0.0, 1.0, 0.0, 0.0)),
      Vectors.dense(Array(1.0, 0.0, 0.0, 0.0))
    ).map(_.toOPVector)

    val splitsRightInclusive = Array(Double.NegativeInfinity, 0.0, 1.0, 5.0, 10.0)
    val expectedRightInclusiveAns = Array(
      Vectors.dense(Array(0.0, 0.0, 0.0, 1.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 1.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 1.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 1.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 0.0)),
      Vectors.dense(Array(0.0, 1.0, 0.0, 0.0)),
      Vectors.dense(Array(1.0, 0.0, 0.0, 0.0))
    ).map(_.toOPVector)

    val trackNullsExpectedAns =
      numbers.zip(expectedAns).map {
        case (None, vec) => vec.value.toArray :+ 1.0
        case (_, vec) => vec.value.toArray :+ 0.0
      }.map(arr => Vectors.dense(arr).toOPVector)
  }

  trait RealTest extends GenericTest {
    lazy val (data1, num) = TestFeatureBuilder("reals", reals)
    val realBucketizer =
      new NumericBucketizer[Real]()
        .setInput(num)
        .setTrackNulls(false)
        .setBuckets(splits, bucketLabels)

    lazy val trackNullsRealBucketizer = realBucketizer.setTrackNulls(true)

    val realBucketizer2 =
      new NumericBucketizer[Real]()
        .setInput(num)
        .setTrackNulls(false)
        .setBuckets(splits)

    lazy val trackNullsRealBucketizer2 = realBucketizer2.setTrackNulls(true)
  }

  trait IntegralTest extends GenericTest {
    lazy val (data1, num) = TestFeatureBuilder("integrals", integrals)

    val integralBucketizer =
      new NumericBucketizer[Integral]()
        .setInput(num)
        .setTrackNulls(false)
        .setBuckets(splits, bucketLabels)

    lazy val trackNullsIntegralBucketizer = integralBucketizer.setTrackNulls(true)
  }

  Spec[NumericBucketizer[_]] should "take a single numeric feature as input and return a vector" in new RealTest {
    val vector = realBucketizer.getOutput()

    vector.name shouldBe realBucketizer.getOutputFeatureName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "validate the params correctly" in new RealTest {
    val Bucketizer3 = new NumericBucketizer[Real]().setInput(num)
    // error because not enough bucket labels
    assertThrows[IllegalArgumentException](
      Bucketizer3.setBuckets(Array[Double](0, 1, 5, 10), Some(Array[String]("0-1K", "1K-5K")))
    )
    // error because split points not in increasing order
    assertThrows[IllegalArgumentException](
      Bucketizer3.setBuckets(Array[Double](10, 1, 5), Some(Array[String]("0-1K", "1K-5K")))
    )
    // error because not >= 3 split points
    assertThrows[IllegalArgumentException](
      Bucketizer3.setBuckets(Array[Double](0, 1), Some(Array[String]("0-1K")))
    )
    // error because NaN split point
    assertThrows[IllegalArgumentException](
      Bucketizer3.setBuckets(Array[Double](0, Double.NaN, 100), Some(Array[String]("0-1K", "1K-5K")))
    )
  }

  it should "update the params correctly" in new RealTest {
    val buck = new NumericBucketizer[Real]().setInput(num)
    val splitz = Array(7.0, 8.0, 10.0, 11.0)
    buck.setBuckets(splitz)
    buck.getSplits shouldBe splitz
    buck.getBucketLabels shouldBe Array("[7.0-8.0)", "[8.0-10.0)", "[10.0-11.0)")
    buck.getSplitInclusion shouldBe Inclusion.Left
    buck.setSplitInclusion(Inclusion.Right)
    buck.getSplits shouldBe splitz
    buck.getBucketLabels shouldBe Array("(7.0-8.0]", "(8.0-10.0]", "(10.0-11.0]")
    buck.getSplitInclusion shouldBe Inclusion.Right
    buck.setBuckets(splitz, bucketLabels = Some(Array("A", "B", "C")))
    buck.getSplits shouldBe splitz
    buck.getBucketLabels shouldBe Array("A", "B", "C")
  }

  it should "throw an exception if the data is out of bounds when trackInvalid is false" in new GenericTest {
    val vals = Seq(Double.PositiveInfinity, Double.NaN, -1, -100).map(_.toReal)
    lazy val (data, num) = TestFeatureBuilder("num", vals)
    val buck = new NumericBucketizer[Real]().setInput(num).setBuckets(splits).setTrackInvalid(false)

    assertThrows[SparkException](buck.transform(data).collect())
  }

  it should "transform the data correctly (reals)" in new RealTest {
    val vector = realBucketizer.getOutput()
    val transformed = realBucketizer.transform(data1)
    val actual = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(actual.head.value.size)(true))
    actual shouldBe expectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      realBucketizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")), IndCol(Some("10-Infinity")))
    )
    OpVectorMetadata(realBucketizer.getOutputFeatureName, realBucketizer.getMetadata()) shouldEqual expectedMeta

    val expectedMeta2 = TestOpVectorMetadataBuilder(
      realBucketizer2,
      num -> List(IndCol(Some("[0.0-1.0)")), IndCol(Some("[1.0-5.0)")),
        IndCol(Some("[5.0-10.0)")), IndCol(Some("[10.0-Infinity)")))
    )
    OpVectorMetadata(realBucketizer2.getOutputFeatureName, realBucketizer2.getMetadata()) shouldEqual expectedMeta2
    val vector2 = realBucketizer2.getOutput()
    val transformed2 = realBucketizer2.transform(data1)
    val actual2 = transformed2.collect(vector2)
    val field2 = transformed2.schema(vector2.name)
    assertNominal(field2, Array.fill(actual2.head.value.size)(true))
  }

  it should "work as a shortcut (reals)" in new RealTest {
    val vector = num.bucketize(trackNulls = false, splits = splits, bucketLabels = bucketLabels)
    vector.originStage shouldBe a[NumericBucketizer[_]]
    val buck = vector.originStage.asInstanceOf[NumericBucketizer[_]]
    val transformed = buck.transform(data1)
    val actual = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(actual.head.value.size)(true))
    actual shouldBe expectedAns
  }

  it should "keep track of null values if wanted (reals) " in new RealTest {
    val vector = trackNullsRealBucketizer.getOutput()
    val transformed = trackNullsRealBucketizer.transform(data1)
    val actual = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(actual.head.value.size)(true))
    actual shouldBe trackNullsExpectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      trackNullsRealBucketizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")), IndCol(Some("10-Infinity")),
        IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(
      trackNullsRealBucketizer.getOutputFeatureName,
      trackNullsRealBucketizer.getMetadata()
    ) shouldEqual expectedMeta

    val expectedMeta2 = TestOpVectorMetadataBuilder(
      trackNullsRealBucketizer2,
      num -> List(IndCol(Some("[0.0-1.0)")), IndCol(Some("[1.0-5.0)")),
        IndCol(Some("[5.0-10.0)")), IndCol(Some("[10.0-Infinity)")),
        IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(
      trackNullsRealBucketizer2.getOutputFeatureName,
      trackNullsRealBucketizer2.getMetadata()
    ) shouldEqual expectedMeta2
    val vector2 = trackNullsRealBucketizer2.getOutput()
    val transformed2 = trackNullsRealBucketizer2.transform(data1)
    val actual2 = transformed2.collect(vector2)
    val field2 = transformed2.schema(vector2.name)
    assertNominal(field2, Array.fill(actual2.head.value.size)(true))
  }

  it should "allow right inclusive splits (reals)" in new RealTest {
    val vector = num.bucketize(trackNulls = false, splits = splitsRightInclusive, splitInclusion = Inclusion.Right)
    val buck = vector.originStage.asInstanceOf[NumericBucketizer[_]]
    val transformed = buck.transform(data1)
    val actual = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(actual.head.value.size)(true))
    actual shouldBe expectedRightInclusiveAns
  }

  it should "correctly bucketize some random reals" in {
    val randomNums = {
      Real(0.0) :: RandomReal.uniform[Real](-10000000.0, 10000000.0)
        .withProbabilityOfEmpty(0.3).limit(1000).filterNot(_.v.exists(_.isNaN))
    }
    val (ds, nums) = TestFeatureBuilder[Real](randomNums)
    val buck = nums.bucketize(
      trackNulls = true,
      splits = Array(Double.NegativeInfinity, 0.0, Double.PositiveInfinity),
      splitInclusion = Inclusion.Left
    )
    val bucketizer = buck.originStage
    val transformed = bucketizer.asInstanceOf[NumericBucketizer[_]].transform(ds)
    val results = transformed.collect(buck)
    val field = transformed.schema(buck.name)
    assertNominal(field, Array.fill(results.head.value.size)(true))

    bucketizer shouldBe a[NumericBucketizer[_]]

    val (neg, pos, empty) =
      (Vectors.dense(1.0, 0.0, 0.0).toOPVector,
        Vectors.dense(0.0, 1.0, 0.0).toOPVector,
        Vectors.dense(0.0, 0.0, 1.0).toOPVector)

    for {(res, exp) <- results.zip(randomNums)}
      withClue(s"Invalid bucket for ${exp.value}: ") {
        res shouldBe (if (exp.isEmpty) empty else if (exp.v.get >= 0.0) pos else neg)
      }
  }

  it should "correctly track invalid values (reals)" in {
    val (ds, num) = TestFeatureBuilder[Real](
      Seq(Real.empty, Double.NaN.toReal, Double.NegativeInfinity.toReal, Double.PositiveInfinity.toReal, Real(10.0))
    )
    val buck = num.bucketize(trackNulls = true, trackInvalid = true, splits = Array(0.0, 1.0, 5.0))
    val stage = buck.originStage.asInstanceOf[NumericBucketizer[_]]
    val transformed = stage.transform(ds)
    val results = transformed.collect(buck)

    results shouldBe Seq(
      Vectors.dense(0.0, 0.0, 0.0, 1.0),
      Vectors.dense(0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 0.0, 1.0, 0.0)
    ).map(_.toOPVector)

    val field = transformed.schema(buck.name)
    assertNominal(field, Array.fill(results.head.value.size)(true))

    val expectedMeta = TestOpVectorMetadataBuilder(
      stage, num -> List(IndCol(Some("[0.0-1.0)")), IndCol(Some("[1.0-5.0)")),
        IndCol(Some(TransmogrifierDefaults.OtherString)), IndCol(Some(TransmogrifierDefaults.NullString))
      )
    )
    OpVectorMetadata(stage.getOutputFeatureName, stage.getMetadata()) shouldEqual expectedMeta
  }

  it should "transform the data correctly (integrals)" in new IntegralTest {
    val vector = integralBucketizer.getOutput()
    val transformed = integralBucketizer.transform(data1)
    val results = transformed.collect(vector)
    results shouldBe expectedAns

    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(results.head.value.size)(true))
    val expectedMeta = TestOpVectorMetadataBuilder(
      integralBucketizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")), IndCol(Some("10-Infinity")))
    )
    OpVectorMetadata(integralBucketizer.getOutputFeatureName, integralBucketizer.getMetadata()) shouldEqual expectedMeta
  }

  it should "work as a shortcut (integrals)" in new IntegralTest {
    val vector = num.bucketize(trackNulls = false, splits = splits, bucketLabels = bucketLabels)
    vector.originStage shouldBe a[NumericBucketizer[_]]
    val buck = vector.originStage.asInstanceOf[NumericBucketizer[_]]
    val transformed = buck.transform(data1)
    val results = transformed.collect(vector)
    results shouldBe expectedAns
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(results.head.value.size)(true))
  }

  it should "keep track of null values if wanted (integrals)" in new IntegralTest {
    val vector = trackNullsIntegralBucketizer.getOutput()
    val transformed = trackNullsIntegralBucketizer.transform(data1)
    val results = transformed.collect(vector)
    results shouldBe trackNullsExpectedAns
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(results.head.value.size)(true))

    val expectedMeta = TestOpVectorMetadataBuilder(
      trackNullsIntegralBucketizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")), IndCol(Some("10-Infinity")),
        IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(
      trackNullsIntegralBucketizer.getOutputFeatureName,
      trackNullsIntegralBucketizer.getMetadata()
    ) shouldEqual expectedMeta
  }

  it should "allow right inclusive splits (integrals)" in new IntegralTest {
    val vector = num.bucketize(trackNulls = false, splits = splitsRightInclusive, splitInclusion = Inclusion.Right)
    val buck = vector.originStage.asInstanceOf[NumericBucketizer[_]]
    val transformed = buck.transform(data1)
    val results = transformed.collect(vector)
    results shouldBe expectedRightInclusiveAns
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(results.head.value.size)(true))

  }

}
