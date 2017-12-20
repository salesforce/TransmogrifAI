/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType.IndCol
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class NumericBucketizerTest extends FlatSpec with TestSparkContext {

  trait GenericTest {
    val numbers = Seq(Some(10.0), None, Some(3.0), Some(5.0), Some(6.0), None, Some(1.0), Some(0.0))
    val reals = numbers.map(n => new Real(n))
    val integrals = numbers.map(n => new Integral(n.map(_.toLong)))

    val buckets = Array(0.0, 1.0, 5.0, 10.0)
    val bucketLabels = Some(Array[String]("0-1", "1-5", "5-10"))
    val expectedAns = Array(
      Vectors.dense(Array(0.0, 0.0, 1.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0)),
      Vectors.dense(Array(0.0, 1.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 1.0)),
      Vectors.dense(Array(0.0, 0.0, 1.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0)),
      Vectors.dense(Array(0.0, 1.0, 0.0)),
      Vectors.dense(Array(1.0, 0.0, 0.0))
    ).map(_.toOPVector)

    val trackNullsExpectedAns = Array(
      Vectors.dense(Array(0.0, 0.0, 1.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 1.0)),
      Vectors.dense(Array(0.0, 1.0, 0.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 1.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 1.0, 0.0)),
      Vectors.dense(Array(0.0, 0.0, 0.0, 1.0)),
      Vectors.dense(Array(0.0, 1.0, 0.0, 0.0)),
      Vectors.dense(Array(1.0, 0.0, 0.0, 0.0))
    ).map(_.toOPVector)
  }

  trait OutOfBounds extends GenericTest {
    lazy val (data1, num) = TestFeatureBuilder("num", Seq(new Real(11.0)))
    lazy val (data2, _) = TestFeatureBuilder(num.name, Seq(new Real(-1)))
    val oobBucketizer = new NumericBucketizer[Double, Real]().setInput(num).setBuckets(buckets)
  }

  trait RealTest extends GenericTest {
    lazy val (data1, num) = TestFeatureBuilder("reals", reals)
    val realBucketizer =
      new NumericBucketizer[Double, Real]()
        .setInput(num)
        .setTrackNulls(false)
        .setBuckets(buckets, bucketLabels)

    lazy val trackNullsRealBucketizer = realBucketizer.setTrackNulls(true)

    val realBucketizer2 =
      new NumericBucketizer[Double, Real]()
        .setInput(num)
        .setTrackNulls(false)
        .setBuckets(buckets)

    lazy val trackNullsRealBucketizer2 = realBucketizer2.setTrackNulls(true)
  }

  trait IntegralTest extends GenericTest {
    lazy val (data1, num) = TestFeatureBuilder("integrals", integrals)

    val integralBucketizer =
      new NumericBucketizer[Long, Integral]()
        .setInput(num)
        .setTrackNulls(false)
        .setBuckets(buckets, bucketLabels)

    lazy val trackNullsIntegralBucketizer = integralBucketizer.setTrackNulls(true)
  }

  Spec[NumericBucketizer[_, _]] should "take a single numeric feature as input and return a vector" in new RealTest {
    val vector = realBucketizer.getOutput()

    vector.name shouldBe realBucketizer.outputName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "validate the params correctly" in new RealTest {
    val Bucketizer3 = new NumericBucketizer[Double, Real]().setInput(num)
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

  it should "throw an exception if the data is out of bounds" in new OutOfBounds {
    assertThrows[Exception](oobBucketizer.transform(data1).collect())
    assertThrows[Exception](oobBucketizer.transform(data2).collect())
  }

  it should "transform the data correctly (reals)" in new RealTest {
    val vector = realBucketizer.getOutput()
    realBucketizer.transform(data1).collect(vector) shouldBe expectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      realBucketizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")))
    )
    OpVectorMetadata(realBucketizer.outputName, realBucketizer.getMetadata()) shouldEqual expectedMeta

    val expectedMeta2 = TestOpVectorMetadataBuilder(
      realBucketizer2,
      num -> List(IndCol(Some("0.0-1.0")), IndCol(Some("1.0-5.0")), IndCol(Some("5.0-10.0")))
    )
    OpVectorMetadata(realBucketizer2.outputName, realBucketizer2.getMetadata()) shouldEqual expectedMeta2
  }

  it should "work as a shortcut (reals)" in new RealTest {
    val vector = num.bucketize(trackNulls = false, splits = buckets, bucketLabels = bucketLabels)
    vector.originStage shouldBe a[NumericBucketizer[_, _]]
    val buck = vector.originStage.asInstanceOf[NumericBucketizer[_, _]]

    buck.transform(data1).collect(vector) shouldBe expectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      buck,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")))
    )
    OpVectorMetadata(buck.outputName, buck.getMetadata()) shouldEqual expectedMeta
  }

  it should "keep track of null values if wanted (reals) " in new RealTest {
    val vector = trackNullsRealBucketizer.getOutput()
    trackNullsRealBucketizer.transform(data1).collect(vector) shouldBe trackNullsExpectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      trackNullsRealBucketizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")),
        IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(trackNullsRealBucketizer.outputName, trackNullsRealBucketizer.getMetadata()) shouldEqual
      expectedMeta

    val expectedMeta2 = TestOpVectorMetadataBuilder(
      trackNullsRealBucketizer2,
      num -> List(IndCol(Some("0.0-1.0")), IndCol(Some("1.0-5.0")), IndCol(Some("5.0-10.0")),
        IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(trackNullsRealBucketizer2.outputName, trackNullsRealBucketizer2.getMetadata()) shouldEqual
      expectedMeta2
  }

  it should "transform the data correctly (integrals)" in new IntegralTest {
    val vector = integralBucketizer.getOutput()
    integralBucketizer.transform(data1).collect(vector) shouldBe expectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      integralBucketizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")))
    )
    OpVectorMetadata(
      integralBucketizer.outputName,
      integralBucketizer.getMetadata()
    ) shouldEqual expectedMeta
  }

  it should "work as a shortcut (integrals)" in new IntegralTest {
    val vector = num.bucketize(trackNulls = false, splits = buckets, bucketLabels = bucketLabels)
    vector.originStage shouldBe a[NumericBucketizer[_, _]]
    val buck = vector.originStage.asInstanceOf[NumericBucketizer[_, _]]
    buck.transform(data1).collect(vector) shouldBe expectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      buck,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")))
    )
    OpVectorMetadata(buck.outputName, buck.getMetadata()) shouldEqual expectedMeta
  }

  it should "keep track of null values if wanted (integrals)" in new IntegralTest {
    val vector = trackNullsIntegralBucketizer.getOutput()
    trackNullsIntegralBucketizer.transform(data1).collect(vector) shouldBe trackNullsExpectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      trackNullsIntegralBucketizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")),
        IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(
      trackNullsIntegralBucketizer.outputName,
      trackNullsIntegralBucketizer.getMetadata()
    ) shouldEqual expectedMeta
  }

}
