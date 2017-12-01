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

  Spec[GeolocationMapVectorizer] should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true)
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.outputName
    vector.parents should contain theSameElementsAs Array(m1, m2)
    vector.originStage shouldBe vectorizer
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return a model that correctly transforms the data" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(18, Array(0, 1, 2, 3, 4, 5, 9, 10, 11),
        Array(32.4, -100.2, 3.0, 33.8, -108.7, 2.0, 45.0, -105.5, 4.0)),
      Vectors.sparse(18, Array(6, 7, 8, 12, 13, 14, 15, 16, 17),
        Array(33.8, -108.7, 2.0, 42.5, -95.4, 4.0, 40.4, -116.3, 2.0)),
      Vectors.sparse(18, Array(), Array())
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
    val vectorizer =
      new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true)
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
    val vectorizer = new GeolocationMapVectorizer().setCleanKeys(true)
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
      m1 -> List(IndVal(Some("A")), IndVal(Some("B"))),
      m2 -> List(IndVal(Some("Z")))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "correctly backlist keys" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true)
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
      m1 -> List(IndVal(Some("B")), IndVal(Some("C"))),
      m2 -> List(IndVal(Some("Y")), IndVal(Some("X")))
    )

    transformed.collect(vector) shouldBe expected
    transformed.schema.toOpVectorMetadata(vectorizer.outputName) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "have a working shortcut function" in {
    val vectorizer = new GeolocationMapVectorizer().setInput(m1, m2).setCleanKeys(true).fit(data)
    val transformed = vectorizer.transform(data)
    val vector = vectorizer.getOutput()
    val expectedOutput = transformed.collect()

    // Now using the shortcut
    val res = m1.vectorize(cleanKeys = Transmogrifier.CleanKeys, others = Array(m2))
    val actualOutput = res.originStage.asInstanceOf[GeolocationMapVectorizer]
      .fit(data).transform(data).collect()

    actualOutput.zip(expectedOutput).forall(f => f._1 == f._2) shouldBe true
  }
}
