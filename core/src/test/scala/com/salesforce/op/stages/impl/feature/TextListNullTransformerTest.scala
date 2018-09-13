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
class TextListNullTransformerTest extends FlatSpec with TestSparkContext with AttributeAsserts {

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

    vector.name shouldBe vectorizer.getOutputFeatureName
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
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected

    val vectorMetadata = vectorizer.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(
      vectorizer,
      f1 -> List(IndCol(Some(TransmogrifierDefaults.NullString))),
      f2 -> List(IndCol(Some(TransmogrifierDefaults.NullString)))
    )
  }
}
