/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.features.Feature
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, RootCol}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class IntegralVectorizerTest extends FlatSpec with TestSparkContext {

  val (testData, inA, inB, inC, inD) = TestFeatureBuilder("inA", "inB", "inC", "inD",
    Seq(
      (Integral(4L), Integral(2L), Integral(2L), Integral.empty),
      (Integral(4L), Integral.empty, Integral(1L), Integral.empty),
      (Integral(2L), Integral(4L), Integral(1L), Integral.empty),
      (Integral.empty, Integral(2L), Integral(2L), Integral.empty),
      (Integral.empty, Integral.empty, Integral.empty, Integral.empty)
    )
  )
  val (testDataDate, inAD, inBD, inCD, inDD) = TestFeatureBuilder("inAD", "inBD", "inCD", "inDD",
    Seq(
      (Date(4L), Date(2L), Date(2L), Date.empty),
      (Date(4L), Date.empty, Date(1L), Date.empty),
      (Date(2L), Date(4L), Date(1L), Date.empty),
      (Date.empty, Date(2L), Date(2L), Date.empty),
      (Date.empty, Date.empty, Date.empty, Date.empty)
    )
  )
  val (testDataDateTime, inADT, inBDT, inCDT, inDDT) = TestFeatureBuilder("inADT", "inBDT", "inCDT", "inDDT",
    Seq(
      (DateTime(4L), DateTime(2L), DateTime(2L), DateTime.empty),
      (DateTime(4L), DateTime.empty, DateTime(1L), DateTime.empty),
      (DateTime(2L), DateTime(4L), DateTime(1L), DateTime.empty),
      (DateTime.empty, DateTime(2L), DateTime(2L), DateTime.empty),
      (DateTime.empty, DateTime.empty, DateTime.empty, DateTime.empty)
    )
  )
  val testVectorizer = new IntegralVectorizer().setInput(inA, inB, inC, inD)
  val outputName = testVectorizer.operationName

  Spec[IntegralVectorizer[_]] should "have output name set correctly" in {
    testVectorizer.operationName shouldBe outputName
  }

  it should "throw an error if you try to get the output without setting the inputs" in {
    intercept[java.util.NoSuchElementException](new IntegralVectorizer[Integral]().getOutput())
  }

  it should "return a single output feature of the correct type" in {
    val outputFeatures = testVectorizer.getOutput()
    outputFeatures shouldBe new Feature[OPVector](
      name = testVectorizer.getOutputFeatureName,
      originStage = testVectorizer,
      isResponse = false,
      parents = Array(inA, inB, inC, inD)
    )
  }

  it should "fit the model with fillWithConstant and transform data correctly" in {
    val testModelConstant = testVectorizer.setFillWithConstant(3L).setTrackNulls(false).fit(testData)

    testModelConstant.parent shouldBe testVectorizer
    testModelConstant.transformFn(Seq(Integral.empty, Integral.empty, Integral.empty)) shouldEqual
      Vectors.dense(3L, 3L, 3L).toOPVector

    val testDataTransformedConstant = testModelConstant.transform(testData)
    val transformedValuesConstant = testDataTransformedConstant.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedConstant.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", "inD", testVectorizer.getOutputFeatureName)

    val expectedZero = Array(
      (4L, 2L, 2L, null, Vectors.dense(4.0, 2.0, 2.0, 3.0)),
      (4L, null, 1L, null, Vectors.dense(4.0, 3.0, 1.0, 3.0)),
      (2L, 4L, 1L, null, Vectors.dense(2.0, 4.0, 1.0, 3.0)),
      (null, 2L, 2L, null, Vectors.dense(3.0, 2.0, 2.0, 3.0)),
      (null, null, null, null, Vectors.dense(3.0, 3.0, 3.0, 3.0))
    )

    transformedValuesConstant.map(_.get(0)) shouldEqual expectedZero.map(_._1)
    transformedValuesConstant.map(_.get(1)) shouldEqual expectedZero.map(_._2)
    transformedValuesConstant.map(_.get(2)) shouldEqual expectedZero.map(_._3)
    transformedValuesConstant.map(_.get(3)) shouldEqual expectedZero.map(_._4)
    transformedValuesConstant.map(_.get(4)) shouldEqual expectedZero.map(_._5)
  }

  it should "fit the model with fillWithMode and transform data correctly" in {
    val testModelMode = testVectorizer.setFillWithMode.setTrackNulls(false).fit(testData)

    testModelMode.parent shouldBe testVectorizer
    testModelMode.transformFn(Seq(Integral.empty, Integral.empty, Integral.empty)) shouldEqual
      Vectors.dense(4.0, 2.0, 1.0).toOPVector

    val testDataTransformedMode = testModelMode.transform(testData)
    val transformedValuesMode = testDataTransformedMode.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMode.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", "inD", testVectorizer.getOutputFeatureName)

    val expectedMode = Array(
      (4.0, 2.0, 2.0, null, Vectors.dense(4.0, 2.0, 2.0, 0.0)),
      (4.0, null, 1.0, null, Vectors.dense(4.0, 2.0, 1.0, 0.0)),
      (2.0, 4.0, 1.0, null, Vectors.dense(2.0, 4.0, 1.0, 0.0)),
      (null, 2.0, 2.0, null, Vectors.dense(4.0, 2.0, 2.0, 0.0)),
      (null, null, null, null, Vectors.dense(4.0, 2.0, 1.0, 0.0))
    )

    transformedValuesMode.map(_.get(0)) shouldEqual expectedMode.map(_._1)
    transformedValuesMode.map(_.get(1)) shouldEqual expectedMode.map(_._2)
    transformedValuesMode.map(_.get(2)) shouldEqual expectedMode.map(_._3)
    transformedValuesMode.map(_.get(3)) shouldEqual expectedMode.map(_._4)
    transformedValuesMode.map(_.get(4)) shouldEqual expectedMode.map(_._5)
  }


  it should "keep track of null values if wanted, using fillWithConstant " in {

    val testModelConstantTracked = testVectorizer.setFillWithConstant(0L).setTrackNulls(true).fit(testData)
    val testDataTransformedConstantTracked = testModelConstantTracked.transform(testData)
    val transformedValuesZeroTracked = testDataTransformedConstantTracked.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedConstantTracked.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", "inD", testVectorizer.getOutputFeatureName)

    val expectedZeroTracked = Array(
      (4.0, 2.0, 2.0, null, Vectors.dense(4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (4.0, null, 1.0, null, Vectors.dense(4.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0)),
      (2.0, 4.0, 1.0, null, Vectors.dense(2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
      (null, 2.0, 2.0, null, Vectors.dense(0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (null, null, null, null, Vectors.dense(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    )

    transformedValuesZeroTracked.map(_.get(0)) shouldEqual expectedZeroTracked.map(_._1)
    transformedValuesZeroTracked.map(_.get(1)) shouldEqual expectedZeroTracked.map(_._2)
    transformedValuesZeroTracked.map(_.get(2)) shouldEqual expectedZeroTracked.map(_._3)
    transformedValuesZeroTracked.map(_.get(3)) shouldEqual expectedZeroTracked.map(_._4)
    transformedValuesZeroTracked.map(_.get(4)) shouldEqual expectedZeroTracked.map(_._5)

    val fieldMetadata = testDataTransformedConstantTracked
      .select(testVectorizer.getOutputFeatureName).schema.fields
      .map(_.metadata).head

    val expectedMeta = TestOpVectorMetadataBuilder(
      testVectorizer,
      inA -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inB -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inC -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inD -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(testVectorizer.getOutputFeatureName, fieldMetadata) shouldBe expectedMeta
  }


  it should "keep track of null values if wanted, using fillWithMode " in {
    val testModelModeTracked = testVectorizer.setFillWithMode.setTrackNulls(true).fit(testData)
    val testDataTransformedModeTracked = testModelModeTracked.transform(testData)
    val transformedValuesModeTracked = testDataTransformedModeTracked.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedModeTracked.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", "inD", testVectorizer.getOutputFeatureName)

    val expectedModeTracked = Array(
      (4.0, 2.0, 2.0, null, Vectors.dense(4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (4.0, null, 1.0, null, Vectors.dense(4.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0)),
      (2.0, 4.0, 1.0, null, Vectors.dense(2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
      (null, 2.0, 2.0, null, Vectors.dense(4.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (null, null, null, null, Vectors.dense(4.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0))
    )

    transformedValuesModeTracked.map(_.get(0)) shouldEqual expectedModeTracked.map(_._1)
    transformedValuesModeTracked.map(_.get(1)) shouldEqual expectedModeTracked.map(_._2)
    transformedValuesModeTracked.map(_.get(2)) shouldEqual expectedModeTracked.map(_._3)
    transformedValuesModeTracked.map(_.get(3)) shouldEqual expectedModeTracked.map(_._4)
    transformedValuesModeTracked.map(_.get(4)) shouldEqual expectedModeTracked.map(_._5)

    val fieldMetadata = testDataTransformedModeTracked
      .select(testVectorizer.getOutputFeatureName).schema.fields
      .map(_.metadata).head
    val expectedMeta = TestOpVectorMetadataBuilder(
      testVectorizer,
      inA -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inB -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inC -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inD -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(testVectorizer.getOutputFeatureName, fieldMetadata) shouldBe expectedMeta
  }

  it should "work the same with DateTime columns as if they were Integral" in {
    val testVectorizer = new IntegralVectorizer().setInput(inADT, inBDT, inCDT, inDDT)
    val testModelMode = testVectorizer.setFillWithMode.setTrackNulls(false).fit(testDataDateTime)

    testModelMode.parent shouldBe testVectorizer
    testModelMode.transformFn(Seq(DateTime.empty, DateTime.empty, DateTime.empty)) shouldEqual
      Vectors.dense(4.0, 2.0, 1.0).toOPVector

    val testDataTransformedMode = testModelMode.transform(testDataDateTime)
    val transformedValuesMode = testDataTransformedMode.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMode.schema.fieldNames should contain theSameElementsAs
      Array("inADT", "inBDT", "inCDT", "inDDT", testVectorizer.getOutputFeatureName)

    val expectedMode = Array(
      (4.0, 2.0, 2.0, null, Vectors.dense(4.0, 2.0, 2.0, 0.0)),
      (4.0, null, 1.0, null, Vectors.dense(4.0, 2.0, 1.0, 0.0)),
      (2.0, 4.0, 1.0, null, Vectors.dense(2.0, 4.0, 1.0, 0.0)),
      (null, 2.0, 2.0, null, Vectors.dense(4.0, 2.0, 2.0, 0.0)),
      (null, null, null, null, Vectors.dense(4.0, 2.0, 1.0, 0.0))
    )

    transformedValuesMode.map(_.get(0)) shouldEqual expectedMode.map(_._1)
    transformedValuesMode.map(_.get(1)) shouldEqual expectedMode.map(_._2)
    transformedValuesMode.map(_.get(2)) shouldEqual expectedMode.map(_._3)
    transformedValuesMode.map(_.get(3)) shouldEqual expectedMode.map(_._4)
    transformedValuesMode.map(_.get(4)) shouldEqual expectedMode.map(_._5)
  }

  it should "correctly vectorize Date columns" in {
    val testVectorizer = new IntegralVectorizer().setInput(inAD, inBD, inCD, inDD)
    val testModelMode = testVectorizer.setFillWithMode.setTrackNulls(false).fit(testDataDate)

    testModelMode.parent shouldBe testVectorizer
    testModelMode.transformFn(Seq(Date.empty, Date.empty, Date.empty)) shouldEqual
      Vectors.dense(4.0, 2.0, 1.0).toOPVector

    val testDataTransformedMode = testModelMode.transform(testDataDate)
    val transformedValuesMode = testDataTransformedMode.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMode.schema.fieldNames should contain theSameElementsAs
      Array("inAD", "inBD", "inCD", "inDD", testVectorizer.getOutputFeatureName)

    val expectedMode = Array(
      (4.0, 2.0, 2.0, null, Vectors.dense(4.0, 2.0, 2.0, 0.0)),
      (4.0, null, 1.0, null, Vectors.dense(4.0, 2.0, 1.0, 0.0)),
      (2.0, 4.0, 1.0, null, Vectors.dense(2.0, 4.0, 1.0, 0.0)),
      (null, 2.0, 2.0, null, Vectors.dense(4.0, 2.0, 2.0, 0.0)),
      (null, null, null, null, Vectors.dense(4.0, 2.0, 1.0, 0.0))
    )

    transformedValuesMode.map(_.get(0)) shouldEqual expectedMode.map(_._1)
    transformedValuesMode.map(_.get(1)) shouldEqual expectedMode.map(_._2)
    transformedValuesMode.map(_.get(2)) shouldEqual expectedMode.map(_._3)
    transformedValuesMode.map(_.get(3)) shouldEqual expectedMode.map(_._4)
    transformedValuesMode.map(_.get(4)) shouldEqual expectedMode.map(_._5)
  }

  it should "correctly vectorize DateTime columns" in {
    val testVectorizer = new IntegralVectorizer().setInput(inADT, inBDT, inCDT, inDDT)
    val testModelMode = testVectorizer.setFillWithMode.setTrackNulls(false).fit(testDataDateTime)

    testModelMode.parent shouldBe testVectorizer
    testModelMode.transformFn(Seq(DateTime.empty, DateTime.empty, DateTime.empty)) shouldEqual
      Vectors.dense(4.0, 2.0, 1.0).toOPVector

    val testDataTransformedMode = testModelMode.transform(testDataDateTime)
    val transformedValuesMode = testDataTransformedMode.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMode.schema.fieldNames should contain theSameElementsAs
      Array("inADT", "inBDT", "inCDT", "inDDT", testVectorizer.getOutputFeatureName)

    val expectedMode = Array(
      (4.0, 2.0, 2.0, null, Vectors.dense(4.0, 2.0, 2.0, 0.0)),
      (4.0, null, 1.0, null, Vectors.dense(4.0, 2.0, 1.0, 0.0)),
      (2.0, 4.0, 1.0, null, Vectors.dense(2.0, 4.0, 1.0, 0.0)),
      (null, 2.0, 2.0, null, Vectors.dense(4.0, 2.0, 2.0, 0.0)),
      (null, null, null, null, Vectors.dense(4.0, 2.0, 1.0, 0.0))
    )

    transformedValuesMode.map(_.get(0)) shouldEqual expectedMode.map(_._1)
    transformedValuesMode.map(_.get(1)) shouldEqual expectedMode.map(_._2)
    transformedValuesMode.map(_.get(2)) shouldEqual expectedMode.map(_._3)
    transformedValuesMode.map(_.get(3)) shouldEqual expectedMode.map(_._4)
    transformedValuesMode.map(_.get(4)) shouldEqual expectedMode.map(_._5)
  }
}
