/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class IntegralMapVectorizerTest extends FlatSpec with TestSparkContext {

  lazy val (data, m1, m2) = TestFeatureBuilder("m1", "m2",
    Seq(
      (Map("a" -> 1L, "b" -> 5L), Map("z" -> 10L)),
      (Map("c" -> 11L), Map("y" -> 3L, "x" -> 0L)),
      (Map.empty[String, Long], Map.empty[String, Long])
    ).map(v => v._1.toIntegralMap -> v._2.toIntegralMap)
  )

  val nullIndicatorValue = Some(OpVectorColumnMetadata.NullString)

  val baseVectorizer = new IntegralMapVectorizer().setInput(m1, m2).setCleanKeys(true)

  val expectedMeta = TestOpVectorMetadataBuilder(
    baseVectorizer,
    m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B"),
      IndColWithGroup(None, "C")),
    m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "X"))
  )


  val expectedMetaTrackNulls = TestOpVectorMetadataBuilder(
    baseVectorizer,
    m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(nullIndicatorValue, "A"),
      IndColWithGroup(None, "B"), IndColWithGroup(nullIndicatorValue, "B"),
      IndColWithGroup(None, "C"), IndColWithGroup(nullIndicatorValue, "C")),
    m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(nullIndicatorValue, "Z"),
      IndColWithGroup(None, "Y"), IndColWithGroup(nullIndicatorValue, "Y"),
      IndColWithGroup(None, "X"), IndColWithGroup(nullIndicatorValue, "X"))
  )

  Spec[IntegralMapVectorizer[_]] should "take an array of features as input and return a single vector feature" in {
    val vector = baseVectorizer.getOutput()

    vector.name shouldBe baseVectorizer.getOutputFeatureName
    vector.parents should contain theSameElementsAs Array(m1, m2)
    vector.originStage shouldBe baseVectorizer
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return a model that correctly transforms the data" in {
    val vectorizer = baseVectorizer.setTrackNulls(false).fit(data)

    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.dense(Array(1.0, 5.0, 0.0, 10.0, 0.0, 0.0)),
      Vectors.sparse(6, Array(2, 4), Array(11.0, 3.0)),
      Vectors.sparse(6, Array(), Array())
    ).map(_.toOPVector)


    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls" in {
    val vectorizer = baseVectorizer.setTrackNulls(true).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.sparse(12, Array(0, 2, 5, 6, 9, 11), Array(1.0, 5.0, 1.0, 10.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(1, 3, 4, 7, 8), Array(1.0, 1.0, 11.0, 1.0, 3.0)),
      Vectors.sparse(12, Array(1, 3, 5, 7, 9, 11), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)


    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMetaTrackNulls
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMetaTrackNulls
  }

  it should "use the correct fill value for missing keys" in {
    val vectorizer = baseVectorizer.setDefaultValue(100).setTrackNulls(false).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.dense(Array(1.0, 5.0, 100.0, 10.0, 100.0, 100.0)),
      Vectors.dense(Array(100.0, 100.0, 11.0, 100.0, 3.0, 0.0)),
      Vectors.dense(Array(100.0, 100.0, 100.0, 100.0, 100.0, 100.0))
    ).map(_.toOPVector)

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with the correct fill value for missing keys" in {
    val vectorizer = baseVectorizer.setDefaultValue(100).setTrackNulls(true).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.sparse(12, Array(0, 2, 4, 5, 6, 8, 9, 10, 11), Array(1.0, 5.0, 100.0, 1.0, 10.0, 100.0, 1.0, 100.0, 1.0)),
      Vectors.sparse(12, Array(0, 1, 2, 3, 4, 6, 7, 8), Array(100.0, 1.0, 100.0, 1.0, 11.0, 100.0, 1.0, 3.0)),
      Vectors.dense(Array(100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0))
    ).map(_.toOPVector)


    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMetaTrackNulls
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMetaTrackNulls
  }

  it should "correctly whitelist keys" in {
    val vectorizer =
      new IntegralMapVectorizer[IntegralMap]().setInput(m1, m2).setCleanKeys(true).setTrackNulls(false)
        .setWhiteListKeys(Array("a", "b", "z")).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.dense(Array(1.0, 5.0, 10.0)),
      Vectors.sparse(3, Array(), Array()),
      Vectors.sparse(3, Array(), Array())
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B")),
      m2 -> List(IndColWithGroup(None, "Z"))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly track nulls with whitelist keys" in {
    val vectorizer =
      new IntegralMapVectorizer[IntegralMap]().setInput(m1, m2).setCleanKeys(true).setTrackNulls(true)
        .setWhiteListKeys(Array("a", "b", "z")).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

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

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly backlist keys" in {
    val vectorizer = new IntegralMapVectorizer[IntegralMap]()
      .setInput(m1, m2).setCleanKeys(true).setTrackNulls(false).setBlackListKeys(Array("a", "z")).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.sparse(4, Array(0), Array(5.0)),
      Vectors.dense(Array(0.0, 11.0, 3.0, 0.0)),
      Vectors.sparse(4, Array(), Array())
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "B"), IndColWithGroup(None, "C")),
      m2 -> List(IndColWithGroup(None, "Y"), IndColWithGroup(None, "X"))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "track nulls with backlist keys" in {
    val vectorizer = new IntegralMapVectorizer[IntegralMap]()
      .setInput(m1, m2).setCleanKeys(true).setTrackNulls(true).setBlackListKeys(Array("a", "z")).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.sparse(8, Array(0, 3, 5, 7), Array(5.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(8, Array(1, 2, 4, 6), Array(1.0, 11.0, 3.0, 0.0)),
      Vectors.sparse(8, Array(1, 3, 5, 7), Array(1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "B"), IndColWithGroup(nullIndicatorValue, "B"),
        IndColWithGroup(None, "C"), IndColWithGroup(nullIndicatorValue, "C")),
      m2 -> List(IndColWithGroup(None, "Y"), IndColWithGroup(nullIndicatorValue, "Y"),
        IndColWithGroup(None, "X"), IndColWithGroup(nullIndicatorValue, "X"))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.getOutputFeatureName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }
}
