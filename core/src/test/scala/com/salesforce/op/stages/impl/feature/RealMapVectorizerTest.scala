/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, IndColWithGroup}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichStructType._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner



@RunWith(classOf[JUnitRunner])
class RealMapVectorizerTest extends FlatSpec with TestSparkContext {

  lazy val (data, m1, m2) = TestFeatureBuilder("m1", "m2",
    Seq(
      (Map("a" -> 1.0, "b" -> 5.0), Map("z" -> 10.0)),
      (Map("c" -> 11.0), Map("y" -> 3.0, "x" -> 0.0)),
      (Map.empty[String, Double], Map.empty[String, Double])
    ).map(v => v._1.toRealMap -> v._2.toRealMap)
  )

  lazy val (meanData, f1, f2) = TestFeatureBuilder("f1", "f2",
    Seq(
      (Map("a" -> 1.0, "b" -> 5.0), Map("y" -> 4.0, "x" -> 0.0, "z" -> 10.0)),
      (Map("a" -> -3.0, "b" -> 3.0, "c" -> 11.0), Map("y" -> 3.0, "x" -> 0.0)),
      (Map.empty[String, Double], Map("y" -> 1.0, "x" -> 0.0, "z" -> 5.0)),
      (Map.empty[String, Double], Map.empty[String, Double])
    ).map(v => v._1.toRealMap -> v._2.toRealMap)
  )

  Spec[RealMapVectorizer[_]] should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setCleanKeys(true).setInput(m1, m2)
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.outputName
    vector.parents should contain theSameElementsAs Array(m1, m2)
    vector.originStage shouldBe vectorizer
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return a model that correctly transforms the data" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2).setDefaultValue(0.0).setCleanKeys(true).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(Array(1.0, 5.0, 0.0, 10.0, 0.0, 0.0)),
      Vectors.sparse(6, Array(2, 4), Array(11.0, 3.0)),
      Vectors.sparse(6, Array(), Array())
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B"), IndColWithGroup(None, "C")),
      m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "X"))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "use the correct fill value for missing keys" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2).setCleanKeys(true)
      .setDefaultValue(100).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(Array(1.0, 5.0, 100.0, 10.0, 100.0, 100.0)),
      Vectors.dense(Array(100.0, 100.0, 11.0, 100.0, 3.0, 0.0)),
      Vectors.dense(Array(100.0, 100.0, 100.0, 100.0, 100.0, 100.0))
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B"), IndColWithGroup(None, "C")),
      m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "X"))
    )
    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly whitelist keys" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2).setDefaultValue(0.0)
      .setCleanKeys(true).setWhiteListKeys(Array("a", "b", "z")).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
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
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly backlist keys" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2).setDefaultValue(0.0)
      .setCleanKeys(true).setBlackListKeys(Array("a", "z")).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
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
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly calculate means by key and fill missing values with them" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(f1, f2).setCleanKeys(true)
      .setFillWithMean(true).fit(meanData)
    val transformed = vectorizer.transform(meanData)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.dense(1.0, 5.0, 11.0, 4.0, 0.0, 10.0),
      Vectors.dense(-3.0, 3.0, 11.0, 3.0, 0.0, 15.0 / 2),
      Vectors.dense(-1.0, 4.0, 11.0, 1.0, 0.0, 5.0),
      Vectors.dense(-1.0, 4.0, 11.0, 8.0 / 3, 0.0, 15.0 / 2)
    ).map(_.toOPVector)

    transformed.collect(vector) shouldBe expected

    // TODO: Order of columns needed to be changed here compared to earlier tests - why does this happen?
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B"), IndColWithGroup(None, "C")),
      f2 -> List(IndColWithGroup(None, "Y"), IndColWithGroup(None, "X"), IndColWithGroup(None, "Z"))
    )
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

}
