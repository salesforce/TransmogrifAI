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
class BinaryVectorizerTest extends FlatSpec with TestSparkContext {

  val (ds, f1, f2) = TestFeatureBuilder(
    Seq[(Binary, Binary)](
      (Binary(false), Binary(false)),
      (Binary(false), Binary(true)),
      (Binary(true), Binary(false)),
      (Binary(true), Binary(true)),
      (Binary.empty, Binary(false)),
      (Binary.empty, Binary(true)),
      (Binary(false), Binary.empty),
      (Binary(true), Binary.empty),
      (Binary.empty, Binary.empty)
    )
  )

  Spec[BinaryVectorizer] should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new BinaryVectorizer().setInput(f1, f2)
    val vector = vectorizer.getOutput()
    vector.name shouldBe vectorizer.outputName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "transform the data correctly [trackNulls=true,fillValue=false]" in {
    val vectorizer = new BinaryVectorizer().setInput(f1, f2).setTrackNulls(true).setFillValue(false)
    val transformed = vectorizer.transform(ds)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Array(0.0, 0.0, 0.0, 0.0),
      Array(0.0, 0.0, 1.0, 0.0),
      Array(1.0, 0.0, 0.0, 0.0),
      Array(1.0, 0.0, 1.0, 0.0),
      Array(0.0, 1.0, 0.0, 0.0),
      Array(0.0, 1.0, 1.0, 0.0),
      Array(0.0, 0.0, 0.0, 1.0),
      Array(1.0, 0.0, 0.0, 1.0),
      Array(0.0, 1.0, 0.0, 1.0)
    ).map(Vectors.dense(_).toOPVector)
    transformed.collect(vector) shouldBe expected
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(RootCol, IndCol(Some(Transmogrifier.NullString))),
      f2 -> List(RootCol, IndCol(Some(Transmogrifier.NullString)))
    )
  }

  it should "transform the data correctly [trackNulls=true,fillValue=true]" in {
    val vectorizer = new BinaryVectorizer().setInput(f1, f2).setTrackNulls(true).setFillValue(true)
    val transformed = vectorizer.transform(ds)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Array(0.0, 0.0, 0.0, 0.0),
      Array(0.0, 0.0, 1.0, 0.0),
      Array(1.0, 0.0, 0.0, 0.0),
      Array(1.0, 0.0, 1.0, 0.0),
      Array(1.0, 1.0, 0.0, 0.0),
      Array(1.0, 1.0, 1.0, 0.0),
      Array(0.0, 0.0, 1.0, 1.0),
      Array(1.0, 0.0, 1.0, 1.0),
      Array(1.0, 1.0, 1.0, 1.0)
    ).map(Vectors.dense(_).toOPVector)
    transformed.collect(vector) shouldBe expected
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(RootCol, IndCol(Some(Transmogrifier.NullString))),
      f2 -> List(RootCol, IndCol(Some(Transmogrifier.NullString)))
    )
  }

  it should "transform the data correctly [trackNulls=false,fillValue=false]" in {
    val vectorizer = new BinaryVectorizer().setInput(f1, f2).setTrackNulls(false).setFillValue(false)
    val transformed = vectorizer.transform(ds)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Array(0.0, 0.0),
      Array(0.0, 1.0),
      Array(1.0, 0.0),
      Array(1.0, 1.0),
      Array(0.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 0.0),
      Array(1.0, 0.0),
      Array(0.0, 0.0)
    ).map(Vectors.dense(_).toOPVector)
    transformed.collect(vector) shouldBe expected
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(RootCol),
      f2 -> List(RootCol)
    )
  }

  it should "transform the data correctly [trackNulls=false,fillValue=true]" in {
    val vectorizer = new BinaryVectorizer().setInput(f1, f2).setTrackNulls(false).setFillValue(true)
    val transformed = vectorizer.transform(ds)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Array(0.0, 0.0),
      Array(0.0, 1.0),
      Array(1.0, 0.0),
      Array(1.0, 1.0),
      Array(1.0, 0.0),
      Array(1.0, 1.0),
      Array(0.0, 1.0),
      Array(1.0, 1.0),
      Array(1.0, 1.0)
    ).map(Vectors.dense(_).toOPVector)
    transformed.collect(vector) shouldBe expected
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(RootCol),
      f2 -> List(RootCol)
    )
  }
}
