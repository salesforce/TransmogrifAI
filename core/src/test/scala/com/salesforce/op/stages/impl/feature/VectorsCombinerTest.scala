/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types.Text
import com.salesforce.op.features.{FeatureLike, TransientFeature}
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.utils.spark.RichMetadata._

@RunWith(classOf[JUnitRunner])
class VectorsCombinerTest extends FlatSpec with PassengerSparkFixtureTest {

  val vectors = Seq(
    Vectors.sparse(4, Array(0, 3), Array(1.0, 1.0)),
    Vectors.dense(Array(2.0, 3.0, 4.0)),
    Vectors.sparse(4, Array(1), Array(777.0))
  )
  val expected = Vectors.sparse(11, Array(0, 3, 4, 5, 6, 8), Array(1.0, 1.0, 2.0, 3.0, 4.0, 777.0))

  Spec[VectorsCombiner] should "combine vectors correctly" in {
    val combined = VectorsCombiner.combine(vectors)
    assert(combined.compressed == combined, "combined is expected to be compressed")
    combined shouldBe expected
  }

  it should "combine metadata correctly" in {
    val vector = Seq(height, description, stringMap).transmogrify()
    val inputs = vector.parents
    val outputData = new OpWorkflow().setReader(dataReader)
      .setResultFeatures(vector, inputs(0), inputs(1), inputs(2))
      .train().score()
    val inputMetadata = OpVectorMetadata.flatten(vector.name,
      inputs.map(i => OpVectorMetadata(outputData.schema(i.name))))
    OpVectorMetadata(outputData.schema(vector.name)).columns should contain theSameElementsAs inputMetadata.columns
  }

  it should "create metadata correctly" in {
    val descVect = description.map[Text]{
      t =>
        Text(t.value match {
          case Some(text) => "this is dumb " + text
          case None => "some STUFF to tokenize"
        })
    }.tokenize().tf(numTerms = 5)
    val vector = Seq(height, stringMap, descVect).transmogrify()
    val Seq(inputs1, inputs2, inputs3) = vector.parents

    val outputData = new OpWorkflow().setReader(dataReader)
      .setResultFeatures(vector, inputs1, inputs2, inputs3)
      .train().score()
    outputData.schema(inputs1.name).metadata.wrapped.getAny("ml_attr").toString shouldBe "Some({\"num_attrs\":5})"

    val inputMetadata = OpVectorMetadata.flatten(vector.name,
      Array(TransientFeature(inputs1).toVectorMetaData(5, Option(inputs1.name)),
        OpVectorMetadata(outputData.schema(inputs2.name)), OpVectorMetadata(outputData.schema(inputs3.name))))
    OpVectorMetadata(outputData.schema(vector.name)).columns should contain theSameElementsAs inputMetadata.columns
  }
}
