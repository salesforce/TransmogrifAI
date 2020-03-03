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
import com.salesforce.op.features.Feature
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, RootCol}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class RealVectorizerTest extends FlatSpec with TestSparkContext with AttributeAsserts {

  val (testData, inA, inB, inC) = TestFeatureBuilder("inA", "inB", "inC",
    Seq[(Real, Real, Real)](
      (Real(4.0), Real(2.0), Real.empty),
      (Real.empty, Real(2.0), Real.empty),
      (Real(2.0), Real.empty, Real.empty)
    )
  )
  val (testDataPercent, inAPer, inBPer, inCPer) = TestFeatureBuilder("inAPer", "inBPer", "inCPer",
    Seq[(Percent, Percent, Percent)](
      (Percent(4.0), Percent(2.0), Percent.empty),
      (Percent.empty, Percent(2.0), Percent.empty),
      (Percent(2.0), Percent.empty, Percent.empty)
    )
  )
  val (testDataCurrency, inACur, inBCur, inCCur) = TestFeatureBuilder("inACur", "inBCur", "inCCur",
    Seq[(Currency, Currency, Currency)](
      (Currency(4.0), Currency(2.0), Currency.empty),
      (Currency.empty, Currency(2.0), Currency.empty),
      (Currency(2.0), Currency.empty, Currency.empty)
    )
  )
  val testVectorizer = new RealVectorizer().setInput(inA, inB, inC)
  val outputName = testVectorizer.operationName

  Spec[RealVectorizer[_]] should "have output name set correctly" in {
    testVectorizer.operationName shouldBe outputName
  }

  it should "throw an error if you try to get the output without setting the inputs" in {
    intercept[java.util.NoSuchElementException](new RealVectorizer[Real]().getOutput())
  }

  it should "return a single output feature of the correct type" in {
    val outputFeatures = testVectorizer.getOutput()
    outputFeatures shouldBe new Feature[OPVector](
      name = testVectorizer.getOutputFeatureName,
      originStage = testVectorizer,
      isResponse = false,
      parents = Array(inA, inB, inC)
    )
  }

  it should "fit the model with fillWithConstant and transform data correctly" in {
    val testModelConstant = testVectorizer.setFillWithConstant(4.2).setTrackNulls(false).fit(testData)

    testModelConstant.parent shouldBe testVectorizer
    testModelConstant.transformFn(Seq(Real.empty, Real.empty, Real.empty)) shouldEqual
      Vectors.dense(4.2, 4.2, 4.2).toOPVector

    val testDataTransformedConstant = testModelConstant.transform(testData)
    val transformedValuesConstant = testDataTransformedConstant.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedConstant.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", testVectorizer.getOutputFeatureName)

    val expectedZero = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 2.0, 4.2)),
      (null, 2.0, null, Vectors.dense(4.2, 2.0, 4.2)),
      (2.0, null, null, Vectors.dense(2.0, 4.2, 4.2))
    )
    val field = testDataTransformedConstant.schema(testModelConstant.getOutputFeatureName)
    assertNominal(field, Array.fill(expectedZero.head._4.size)(false),
      testDataTransformedConstant.collect(testModelConstant.getOutput()))
    transformedValuesConstant.map(_.get(0)) shouldEqual expectedZero.map(_._1)
    transformedValuesConstant.map(_.get(1)) shouldEqual expectedZero.map(_._2)
    transformedValuesConstant.map(_.get(2)) shouldEqual expectedZero.map(_._3)
    transformedValuesConstant.map(_.get(3)) shouldEqual expectedZero.map(_._4)
  }

  it should "fit the model with fillWithMean and transform data correctly" in {
    val testModelMean = testVectorizer.setFillWithMean.setTrackNulls(false).fit(testData)

    testModelMean.parent shouldBe testVectorizer
    testModelMean.transformFn(Seq(Real.empty, Real.empty, Real.empty)) shouldEqual
      Vectors.dense(3.0, 2.0, 0.0).toOPVector

    val testDataTransformedMean = testModelMean.transform(testData)
    val transformedValuesMean = testDataTransformedMean.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMean.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", testVectorizer.getOutputFeatureName)

    val expectedMean = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 2.0, 0.0)),
      (null, 2.0, null, Vectors.dense(3.0, 2.0, 0.0)),
      (2.0, null, null, Vectors.dense(2.0, 2.0, 0.0))
    )
    val field = testDataTransformedMean.schema(testModelMean.getOutputFeatureName)
    assertNominal(field, Array.fill(expectedMean.head._4.size)(false),
      testDataTransformedMean.collect(testModelMean.getOutput()))
    transformedValuesMean.map(_.get(0)) shouldEqual expectedMean.map(_._1)
    transformedValuesMean.map(_.get(1)) shouldEqual expectedMean.map(_._2)
    transformedValuesMean.map(_.get(2)) shouldEqual expectedMean.map(_._3)
    transformedValuesMean.map(_.get(3)) shouldEqual expectedMean.map(_._4)
  }

  it should "keep track of null values if wanted, using fillWithConstant " in {

    val testModelConstantTracked = testVectorizer.setFillWithConstant(0.0).setTrackNulls(true).fit(testData)
    val testDataTransformedConstantTracked = testModelConstantTracked.transform(testData)
    val transformedValuesZeroTracked = testDataTransformedConstantTracked.collect()
    // This is string because of vector type being private to spark ml
    testDataTransformedConstantTracked.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", testVectorizer.getOutputFeatureName)

    val expectedZeroTracked = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (null, 2.0, null, Vectors.dense(0.0, 1.0, 2.0, 0.0, 0.0, 1.0)),
      (2.0, null, null, Vectors.dense(2.0, 0.0, 0.0, 1.0, 0.0, 1.0))
    )
    val field = testDataTransformedConstantTracked.schema(testModelConstantTracked.getOutputFeatureName)
    assertNominal(field, Array.fill(expectedZeroTracked.head._4.size / 2)(Seq(false, true)).flatten,
      testDataTransformedConstantTracked.collect(testModelConstantTracked.getOutput()))
    transformedValuesZeroTracked.map(_.get(0)) shouldEqual expectedZeroTracked.map(_._1)
    transformedValuesZeroTracked.map(_.get(1)) shouldEqual expectedZeroTracked.map(_._2)
    transformedValuesZeroTracked.map(_.get(2)) shouldEqual expectedZeroTracked.map(_._3)
    transformedValuesZeroTracked.map(_.get(3)) shouldEqual expectedZeroTracked.map(_._4)

    val fieldMetadata = testDataTransformedConstantTracked
      .select(testVectorizer.getOutputFeatureName).schema.fields
      .map(_.metadata).head

    val expectedMeta = TestOpVectorMetadataBuilder(
      testVectorizer,
      inA -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inB -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inC -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(testVectorizer.getOutputFeatureName, fieldMetadata) shouldBe expectedMeta
  }

  it should "keep track of null values if wanted, using fillWithMean" in {
    val testModelMeanTracked = testVectorizer.setFillWithMean.setTrackNulls(true).fit(testData)
    val testDataTransformedMeanTracked = testModelMeanTracked.transform(testData)
    val transformedValuesMeanTracked = testDataTransformedMeanTracked.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMeanTracked.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", testVectorizer.getOutputFeatureName)

    val expectedMeanTracked = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (null, 2.0, null, Vectors.dense(3.0, 1.0, 2.0, 0.0, 0.0, 1.0)),
      (2.0, null, null, Vectors.dense(2.0, 0.0, 2.0, 1.0, 0.0, 1.0))
    )
    val field = testDataTransformedMeanTracked.schema(testModelMeanTracked.getOutputFeatureName)
    assertNominal(field, Array.fill(expectedMeanTracked.head._4.size / 2)(Seq(false, true)).flatten,
      testDataTransformedMeanTracked.collect(testModelMeanTracked.getOutput()))
    transformedValuesMeanTracked.map(_.get(0)) shouldEqual expectedMeanTracked.map(_._1)
    transformedValuesMeanTracked.map(_.get(1)) shouldEqual expectedMeanTracked.map(_._2)
    transformedValuesMeanTracked.map(_.get(2)) shouldEqual expectedMeanTracked.map(_._3)
    transformedValuesMeanTracked.map(_.get(3)) shouldEqual expectedMeanTracked.map(_._4)

    val fieldMetadata = testDataTransformedMeanTracked
      .select(testVectorizer.getOutputFeatureName).schema.fields
      .map(_.metadata).head
    val expectedMeta = TestOpVectorMetadataBuilder(
      testVectorizer,
      inA -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inB -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inC -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(testVectorizer.getOutputFeatureName, fieldMetadata) shouldBe expectedMeta
  }

  it should "work with columns of type Percent just as if they were Real" in {
    val testVectorizer = new RealVectorizer().setInput(inAPer, inBPer, inCPer)
    val testModelMeanTracked = testVectorizer.setFillWithMean.setTrackNulls(true).fit(testDataPercent)
    val testDataTransformedMeanTracked = testModelMeanTracked.transform(testDataPercent)
    val transformedValuesMeanTracked = testDataTransformedMeanTracked.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMeanTracked.schema.fieldNames should contain theSameElementsAs
      Array("inAPer", "inBPer", "inCPer", testVectorizer.getOutputFeatureName)

    val expectedMeanTracked = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (null, 2.0, null, Vectors.dense(3.0, 1.0, 2.0, 0.0, 0.0, 1.0)),
      (2.0, null, null, Vectors.dense(2.0, 0.0, 2.0, 1.0, 0.0, 1.0))
    )
    val field = testDataTransformedMeanTracked.schema(testModelMeanTracked.getOutputFeatureName)
    assertNominal(field, Array.fill(expectedMeanTracked.head._4.size / 2)(Seq(false, true)).flatten,
      testDataTransformedMeanTracked.collect(testModelMeanTracked.getOutput()))
    transformedValuesMeanTracked.map(_.get(0)) shouldEqual expectedMeanTracked.map(_._1)
    transformedValuesMeanTracked.map(_.get(1)) shouldEqual expectedMeanTracked.map(_._2)
    transformedValuesMeanTracked.map(_.get(2)) shouldEqual expectedMeanTracked.map(_._3)
    transformedValuesMeanTracked.map(_.get(3)) shouldEqual expectedMeanTracked.map(_._4)

    val fieldMetadata = testDataTransformedMeanTracked
      .select(testVectorizer.getOutputFeatureName).schema.fields
      .map(_.metadata).head
    val expectedMeta = TestOpVectorMetadataBuilder(
      testVectorizer,
      inAPer -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inBPer -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString))),
      inCPer -> List(RootCol, IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(testVectorizer.getOutputFeatureName, fieldMetadata) shouldBe expectedMeta
  }

  it should "work on columns of type Currency" in {
    val testVectorizer = new RealVectorizer().setInput(inACur, inBCur, inCCur)
    val testModelConstant = testVectorizer.setFillWithConstant(4.2).setTrackNulls(false).fit(testDataCurrency)
    val testDataTransformedConstant = testModelConstant.transform(testDataCurrency)

    testDataTransformedConstant.schema.fieldNames should contain theSameElementsAs
      Array("inACur", "inBCur", "inCCur", testVectorizer.getOutputFeatureName)

    val transformedValuesConstant = testDataTransformedConstant.collect()
    val expectedZero = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 2.0, 4.2)),
      (null, 2.0, null, Vectors.dense(4.2, 2.0, 4.2)),
      (2.0, null, null, Vectors.dense(2.0, 4.2, 4.2))
    )
    val field = testDataTransformedConstant.schema(testModelConstant.getOutputFeatureName)
    assertNominal(field, Array.fill(expectedZero.head._4.size)(false),
      testDataTransformedConstant.collect(testModelConstant.getOutput()))
    transformedValuesConstant.map(_.get(0)) shouldEqual expectedZero.map(_._1)
    transformedValuesConstant.map(_.get(1)) shouldEqual expectedZero.map(_._2)
    transformedValuesConstant.map(_.get(2)) shouldEqual expectedZero.map(_._3)
    transformedValuesConstant.map(_.get(3)) shouldEqual expectedZero.map(_._4)
  }

  it should "work on columns of type Percent" in {
    val testVectorizer = new RealVectorizer().setInput(inAPer, inBPer, inCPer)
    val testModelConstant = testVectorizer.setFillWithConstant(4.2).setTrackNulls(false).fit(testDataPercent)
    val testDataTransformedConstant = testModelConstant.transform(testDataPercent)

    testDataTransformedConstant.schema.fieldNames should contain theSameElementsAs
      Array("inAPer", "inBPer", "inCPer", testVectorizer.getOutputFeatureName)

    val transformedValuesConstant = testDataTransformedConstant.collect()
    val expectedZero = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 2.0, 4.2)),
      (null, 2.0, null, Vectors.dense(4.2, 2.0, 4.2)),
      (2.0, null, null, Vectors.dense(2.0, 4.2, 4.2))
    )
    val field = testDataTransformedConstant.schema(testModelConstant.getOutputFeatureName)
    assertNominal(field, Array.fill(expectedZero.head._4.size)(false),
      testDataTransformedConstant.collect(testModelConstant.getOutput()))
    transformedValuesConstant.map(_.get(0)) shouldEqual expectedZero.map(_._1)
    transformedValuesConstant.map(_.get(1)) shouldEqual expectedZero.map(_._2)
    transformedValuesConstant.map(_.get(2)) shouldEqual expectedZero.map(_._3)
    transformedValuesConstant.map(_.get(3)) shouldEqual expectedZero.map(_._4)
  }

  it should "work on an empty dataset" in {
    val (testData, inA, inB, inC) = TestFeatureBuilder("inA", "inB", "inC", Seq.empty[(Real, Real, Real)])

    val testModel = testVectorizer.fit(testData)
    val testDataTransformed = testModel.transform(testData)

    val expected = Array.empty[Row]

    expected shouldBe testDataTransformed.collect(testModel.getOutput())

  }

  it should "return a new column values for isMin" in {
    val testModel = testVectorizer.setTrackMins(true).fit(testData)
    val testDataTransformed = testModel.transform(testData)

    val actual = testDataTransformed.schema(testModel.getOutput())
  }
}
