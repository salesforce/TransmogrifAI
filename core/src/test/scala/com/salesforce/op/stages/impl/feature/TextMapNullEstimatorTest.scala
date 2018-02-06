/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types.{TextMap, _}
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, IndColWithGroup}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class TextMapNullEstimatorTest extends FlatSpec with TestSparkContext {

  val (ds, f1) = TestFeatureBuilder(
    Seq[(TextMap)](
      TextMap(Map("k1" -> "A giraffe drinks by the watering hole", "k2" -> "Cheese")),
      TextMap(Map("k2" -> "French Fries")),
      TextMap(Map("k3" -> "Hip-hop Pottamus"))
    )
  )

  Spec[TextListNullTransformer[_]] should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new TextMapNullEstimator[TextMap]().setInput(f1)
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.outputName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "transform the data correctly" in {
    val vectorizer = new TextMapNullEstimator[TextMap]().setInput(f1)
    val transformed = vectorizer.fit(ds).transform(ds)
    val vector = vectorizer.getOutput()

    val expected = Array(
      Array(0.0, 0.0, 1.0),
      Array(1.0, 0.0, 1.0),
      Array(1.0, 1.0, 0.0)
    ).map(Vectors.dense(_).toOPVector)
    transformed.collect(vector) shouldBe expected

    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(
        IndColWithGroup(name = Option(TransmogrifierDefaults.NullString), groupName = "k1"),
        IndColWithGroup(name = Option(TransmogrifierDefaults.NullString), groupName = "k2"),
        IndColWithGroup(name = Option(TransmogrifierDefaults.NullString), groupName = "k3")
      )
    )
  }
}
