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
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}



@RunWith(classOf[JUnitRunner])
class RealMapVectorizerTest extends FlatSpec with TestSparkContext {

  lazy val (data, m1, m2) = TestFeatureBuilder("m1", "m2",
    Seq(
      (Map("a" -> 1.0, "b" -> 5.0), Map("z" -> 10.0)),
      (Map("c" -> 11.0), Map("y" -> 3.0, "x" -> 0.0)),
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
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2).setCleanKeys(true).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
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
      m1 -> List(IndVal(Some("A")), IndVal(Some("B")), IndVal(Some("C"))),
      m2 -> List(IndVal(Some("Z")), IndVal(Some("Y")), IndVal(Some("X")))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly whitelist keys" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2)
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
      m1 -> List(IndVal(Some("A")), IndVal(Some("B"))),
      m2 -> List(IndVal(Some("Z")))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly backlist keys" in {
    val vectorizer = new RealMapVectorizer[RealMap]().setInput(m1, m2)
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
      m1 -> List(IndVal(Some("B")), IndVal(Some("C"))),
      m2 -> List(IndVal(Some("Y")), IndVal(Some("X")))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }
}
