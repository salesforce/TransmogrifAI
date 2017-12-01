/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType.IndVal
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
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

  Spec[IntegralMapVectorizer[_]] should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new IntegralMapVectorizer().setInput(m1, m2).setCleanKeys(true)
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.outputName
    vector.parents should contain theSameElementsAs Array(m1, m2)
    vector.originStage shouldBe vectorizer
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return a model that correctly transforms the data" in {
    val vectorizer = new IntegralMapVectorizer[IntegralMap]().setInput(m1, m2).setCleanKeys(true).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.dense(Array(1.0, 5.0, 0.0, 10.0, 0.0, 0.0)),
      Vectors.sparse(6, Array(2, 4), Array(11.0, 3.0)),
      Vectors.sparse(6, Array(), Array())
    ).map(_.toOPVector)

    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndVal(Some("A")), IndVal(Some("B")), IndVal(Some("C"))),
      m2 -> List(IndVal(Some("Z")), IndVal(Some("Y")), IndVal(Some("X")))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    println(s"vectorMetadata: $vectorMetadata")
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "use the correct fill value for missing keys" in {
    val vectorizer = new IntegralMapVectorizer[IntegralMap]().setInput(m1, m2).setCleanKeys(true)
      .setDefaultValue(100).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.dense(Array(1.0, 5.0, 100.0, 10.0, 100.0, 100.0)),
      Vectors.dense(Array(100.0, 100.0, 11.0, 100.0, 3.0, 0.0)),
      Vectors.dense(Array(100.0, 100.0, 100.0, 100.0, 100.0, 100.0))
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndVal(Some("A")), IndVal(Some("B")), IndVal(Some("C"))),
      m2 -> List(IndVal(Some("Z")), IndVal(Some("Y")), IndVal(Some("X")))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly whitelist keys" in {
    val vectorizer =
      new IntegralMapVectorizer[IntegralMap]().setInput(m1, m2).setCleanKeys(true)
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
      m1 -> List(IndVal(Some("A")), IndVal(Some("B"))),
      m2 -> List(IndVal(Some("Z")))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly backlist keys" in {
    val vectorizer = new IntegralMapVectorizer[IntegralMap]()
      .setInput(m1, m2).setCleanKeys(true).setBlackListKeys(Array("a", "z")).fit(data)
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(data)

    val expected = Array(
      Vectors.sparse(4, Array(0), Array(5.0)),
      Vectors.dense(Array(0.0, 11.0, 3.0, 0.0)),
      Vectors.sparse(4, Array(), Array())
    ).map(_.toOPVector)
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndVal(Some("B")), IndVal(Some("C"))),
      m2 -> List(IndVal(Some("Y")), IndVal(Some("X")))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }
}
