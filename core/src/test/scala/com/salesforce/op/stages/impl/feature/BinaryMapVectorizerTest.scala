/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, IndColWithGroup, IndVal}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class BinaryMapVectorizerTest extends FlatSpec with TestSparkContext {

  lazy val (data, m1, m2) = TestFeatureBuilder("m1", "m2",
    Seq(
      (Map("a" -> false, "b" -> true), Map("z" -> false)),
      (Map("c" -> false), Map("y" -> true, "x" -> true)),
      (Map.empty[String, Boolean], Map.empty[String, Boolean])
    ).map(v => v._1.toBinaryMap -> v._2.toBinaryMap)
  )
  val vectorizer = new BinaryMapVectorizer().setInput(m1, m2).setCleanKeys(true)

  /**
   * Note that defaults and filters are tested in [[RealMapVectorizerTest]]
   * as that code is shared between the two classes
   */
  Spec[BinaryMapVectorizer[_]] should "take an array of features as input and return a single vector feature" in {
    val vector = vectorizer.getOutput()

    vector.name shouldBe vectorizer.outputName
    vector.parents should contain theSameElementsAs Array(m1, m2)
    vector.originStage shouldBe vectorizer
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return a model that correctly transforms the data" in {
    val transformed = vectorizer.fit(data).transform(data)
    val vector = vectorizer.getOutput()
    val expected = Array(
      Vectors.sparse(6, Array(1), Array(1.0)),
      Vectors.sparse(6, Array(4, 5), Array(1.0, 1.0)),
      Vectors.sparse(6, Array(), Array())
    ).map(_.toOPVector)

    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      m1 -> List(IndColWithGroup(None, "A"), IndColWithGroup(None, "B"), IndColWithGroup(None, "C")),
      m2 -> List(IndColWithGroup(None, "Z"), IndColWithGroup(None, "Y"), IndColWithGroup(None, "X"))
    )

    transformed.collect(vector) shouldBe expected
    val field = transformed.schema(vectorizer.outputName)
    OpVectorMetadata(field) shouldEqual expectedMeta
    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(field.copy(metadata = vectorMetadata)) shouldEqual expectedMeta
  }

}
