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
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types.{Text, _}
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.{OpEstimatorSpec, PassengerSparkFixtureTest, TestFeatureBuilder}
import com.salesforce.op.testkit.{RandomReal, RandomVector}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class VectorsCombinerTest
  extends OpEstimatorSpec[OPVector, SequenceModel[OPVector, OPVector], VectorsCombiner]
    with PassengerSparkFixtureTest {

  override def specName = classOf[VectorsCombiner].getSimpleName

  val (inputData, f1, f2) = TestFeatureBuilder(Seq(
    Vectors.sparse(4, Array(0, 3), Array(1.0, 1.0)).toOPVector ->
      Vectors.sparse(4, Array(0, 3), Array(2.0, 3.0)).toOPVector,
    Vectors.dense(Array(2.0, 3.0, 4.0)).toOPVector ->
      Vectors.dense(Array(12.0, 13.0, 14.0)).toOPVector,
    // Purposely added some very large sparse vectors to verify the efficiency
    Vectors.sparse(100000000, Array(1), Array(777.0)).toOPVector ->
      Vectors.sparse(500000000, Array(0), Array(888.0)).toOPVector
  ))

  val estimator = new VectorsCombiner().setInput(f1, f2)

  val expectedResult = Seq(
    Vectors.sparse(8, Array(0, 3, 4, 7), Array(1.0, 1.0, 2.0, 3.0)).toOPVector,
    Vectors.dense(Array(2.0, 3.0, 4.0, 12.0, 13.0, 14.0)).toOPVector,
    Vectors.sparse(600000000, Array(1, 100000000), Array(777.0, 888.0)).toOPVector
  )

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
    val descVect = description.map[Text] { t =>
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
