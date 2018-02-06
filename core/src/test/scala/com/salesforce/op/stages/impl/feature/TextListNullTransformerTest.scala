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
class TextListNullTransformerTest extends FlatSpec with TestSparkContext {

  val (ds, f1, f2) = TestFeatureBuilder(
    Seq[(TextList, TextList)](
      (TextList(Seq("A giraffe drinks by the watering hole")), TextList(Seq("A giraffe drinks by the watering hole"))),
      (TextList(Seq("A giraffe drinks by the watering hole")), TextList(Seq("Cheese"))),
      (TextList(Seq("Cheese")), TextList(Seq("A giraffe drinks by the watering hole"))),
      (TextList(Seq("Cheese")), TextList(Seq("Cheese"))),
      (TextList.empty, TextList(Seq("A giraffe drinks by the watering hole"))),
      (TextList.empty, TextList(Seq("Cheese"))),
      (TextList(Seq("A giraffe drinks by the watering hole")), TextList.empty),
      (TextList(Seq("Cheese")), TextList.empty),
      (TextList.empty, TextList.empty)
    )
  )

  Spec[TextListNullTransformer[_]] should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new TextListNullTransformer().setInput(f1, f2)
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.outputName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "transform the data correctly" in {
    val vectorizer = new TextListNullTransformer().setInput(f1, f2)
    val transformed = vectorizer.transform(ds)
    val vector = vectorizer.getOutput()

    val expected = Array(
      Array(0.0, 0.0),
      Array(0.0, 0.0),
      Array(0.0, 0.0),
      Array(0.0, 0.0),
      Array(1.0, 0.0),
      Array(1.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 1.0),
      Array(1.0, 1.0)
    ).map(Vectors.dense(_).toOPVector)
    transformed.collect(vector) shouldBe expected

    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(IndCol(Some(TransmogrifierDefaults.NullString))),
      f2 -> List(IndCol(Some(TransmogrifierDefaults.NullString)))
    )
  }
}
