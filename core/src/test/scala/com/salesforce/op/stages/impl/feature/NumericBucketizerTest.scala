/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, RootCol}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class NumericBucketizerTest extends FlatSpec with TestSparkContext {

  trait GenericTest {

    val numbers = Seq(Some(10.0), None, Some(3.0), Some(5.0), Some(6.0), None, Some(1.0), Some(0.0))
    val reals = numbers.map(n => new Real(n))
    val integrals = numbers.map(n => new Integral(n.map(_.toLong)))

    val buckets = Array[Double](0, 1, 5, 10)
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
  }

  trait OutOfBounds extends GenericTest {
    lazy val (data1, num) = TestFeatureBuilder("num", Seq(new Real(11.0)))
    lazy val (data2, _) = TestFeatureBuilder(num.name, Seq(new Real(-1)))
    val oobVectorizer =
      new NumericBucketizer[Double, Real]()
        .setInput(num)
        .setBuckets(splitPoints = buckets)

  }

  trait RealTest extends GenericTest {
    lazy val (data1, num) = TestFeatureBuilder("reals", reals)

    val realVectorizer =
      new NumericBucketizer[Double, Real]()
        .setInput(num)
        .setBuckets(splitPoints = buckets, bucketLabels = bucketLabels)

    val realVectorizer2 =
      new NumericBucketizer[Double, Real]()
        .setInput(num)
        .setBuckets(splitPoints = buckets)
  }

  trait IntegralTest extends GenericTest {
    lazy val (data1, num) = TestFeatureBuilder("integrals", integrals)

    val integralVectorizer =
      new NumericBucketizer[Long, Integral]()
        .setInput(num)
        .setBuckets(splitPoints = buckets, bucketLabels = bucketLabels)
  }

  Spec[NumericBucketizer[_, _]] should
    "take a single numeric feature as input and return a vector output" in new RealTest {
    val vector = realVectorizer.getOutput()

    vector.name shouldBe realVectorizer.outputName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "transform the data correctly" in new RealTest {
    val vector = realVectorizer.getOutput()
    realVectorizer.transform(data1).collect(vector) shouldBe expectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      realVectorizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")))
    )
    OpVectorMetadata(realVectorizer.outputName, realVectorizer.getMetadata()) shouldEqual expectedMeta

    val expectedMeta2 = TestOpVectorMetadataBuilder(
      realVectorizer2,
      num -> List(IndCol(Some("0.0-1.0")), IndCol(Some("1.0-5.0")), IndCol(Some("5.0-10.0")))
    )
    OpVectorMetadata(realVectorizer2.outputName, realVectorizer2.getMetadata()) shouldEqual expectedMeta2
  }

  it should "validate the params correctly" in new RealTest {
    val vectorizer3 = new NumericBucketizer[Double, Real]().setInput(num)
    // error because not enough bucket labels
    assertThrows[IllegalArgumentException](
      vectorizer3.setBuckets(splitPoints = Array[Double](0, 1, 5, 10),
        bucketLabels = Some(Array[String]("0-1K", "1K-5K")))
    )
    // error because split points not in increasing order
    assertThrows[IllegalArgumentException](
      vectorizer3.setBuckets(splitPoints = Array[Double](10, 1, 5),
        bucketLabels = Some(Array[String]("0-1K", "1K-5K")))
    )
    // error because not >= 3 split points
    assertThrows[IllegalArgumentException](
      vectorizer3.setBuckets(splitPoints = Array[Double](0, 1),
        bucketLabels = Some(Array[String]("0-1K")))
    )
    // error because NaN split point
    assertThrows[IllegalArgumentException](
      vectorizer3.setBuckets(splitPoints = Array[Double](0, Double.NaN, 100),
        bucketLabels = Some(Array[String]("0-1K", "1K-5K")))
    )
  }

  it should "throw an exception if the data is out of bounds" in new OutOfBounds {
    assertThrows[Exception](oobVectorizer.transform(data1).collect())
    assertThrows[Exception](oobVectorizer.transform(data2).collect())
  }

  it should "behave bucket integral values properly" in new IntegralTest {
    val vector = integralVectorizer.getOutput()
    integralVectorizer.transform(data1).collect(vector) shouldBe expectedAns

    val expectedMeta = TestOpVectorMetadataBuilder(
      integralVectorizer,
      num -> List(IndCol(Some("0-1")), IndCol(Some("1-5")), IndCol(Some("5-10")))
    )
    OpVectorMetadata(integralVectorizer.outputName, integralVectorizer.getMetadata()) shouldEqual expectedMeta
  }
}
