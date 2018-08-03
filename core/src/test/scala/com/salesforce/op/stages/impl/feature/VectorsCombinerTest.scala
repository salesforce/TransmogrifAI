/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
