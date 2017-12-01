/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.features.Feature
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, RootCol}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class RealVectorizerTest extends FlatSpec with TestSparkContext {

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
      name = testVectorizer.outputName,
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
      Array("inA", "inB", "inC", testVectorizer.outputName)

    val expectedZero = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 2.0, 4.2)),
      (null, 2.0, null, Vectors.dense(4.2, 2.0, 4.2)),
      (2.0, null, null, Vectors.dense(2.0, 4.2, 4.2))
    )

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
      Array("inA", "inB", "inC", testVectorizer.outputName)

    val expectedMean = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 2.0, 0.0)),
      (null, 2.0, null, Vectors.dense(3.0, 2.0, 0.0)),
      (2.0, null, null, Vectors.dense(2.0, 2.0, 0.0))
    )

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
      Array("inA", "inB", "inC", testVectorizer.outputName)

    val expectedZeroTracked = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (null, 2.0, null, Vectors.dense(0.0, 1.0, 2.0, 0.0, 0.0, 1.0)),
      (2.0, null, null, Vectors.dense(2.0, 0.0, 0.0, 1.0, 0.0, 1.0))
    )

    transformedValuesZeroTracked.map(_.get(0)) shouldEqual expectedZeroTracked.map(_._1)
    transformedValuesZeroTracked.map(_.get(1)) shouldEqual expectedZeroTracked.map(_._2)
    transformedValuesZeroTracked.map(_.get(2)) shouldEqual expectedZeroTracked.map(_._3)
    transformedValuesZeroTracked.map(_.get(3)) shouldEqual expectedZeroTracked.map(_._4)

    val fieldMetadata = testDataTransformedConstantTracked
      .select(testVectorizer.outputName).schema.fields
      .map(_.metadata).head

    val expectedMeta = TestOpVectorMetadataBuilder(
      testVectorizer,
      inA -> List(RootCol, IndCol(Some(Transmogrifier.NullString))),
      inB -> List(RootCol, IndCol(Some(Transmogrifier.NullString))),
      inC -> List(RootCol, IndCol(Some(Transmogrifier.NullString)))
    )
    OpVectorMetadata(testVectorizer.outputName, fieldMetadata) shouldBe expectedMeta
  }

  it should "keep track of null values if wanted, using fillWithMean" in {
    val testModelMeanTracked = testVectorizer.setFillWithMean.setTrackNulls(true).fit(testData)
    val testDataTransformedMeanTracked = testModelMeanTracked.transform(testData)
    val transformedValuesMeanTracked = testDataTransformedMeanTracked.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMeanTracked.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", testVectorizer.outputName)

    val expectedMeanTracked = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (null, 2.0, null, Vectors.dense(3.0, 1.0, 2.0, 0.0, 0.0, 1.0)),
      (2.0, null, null, Vectors.dense(2.0, 0.0, 2.0, 1.0, 0.0, 1.0))
    )

    transformedValuesMeanTracked.map(_.get(0)) shouldEqual expectedMeanTracked.map(_._1)
    transformedValuesMeanTracked.map(_.get(1)) shouldEqual expectedMeanTracked.map(_._2)
    transformedValuesMeanTracked.map(_.get(2)) shouldEqual expectedMeanTracked.map(_._3)
    transformedValuesMeanTracked.map(_.get(3)) shouldEqual expectedMeanTracked.map(_._4)

    val fieldMetadata = testDataTransformedMeanTracked
      .select(testVectorizer.outputName).schema.fields
      .map(_.metadata).head
    val expectedMeta = TestOpVectorMetadataBuilder(
      testVectorizer,
      inA -> List(RootCol, IndCol(Some(Transmogrifier.NullString))),
      inB -> List(RootCol, IndCol(Some(Transmogrifier.NullString))),
      inC -> List(RootCol, IndCol(Some(Transmogrifier.NullString)))
    )
    OpVectorMetadata(testVectorizer.outputName, fieldMetadata) shouldBe expectedMeta
  }

  it should "work with columns of type Percent just as if they were Real" in {
    val testVectorizer = new RealVectorizer().setInput(inAPer, inBPer, inCPer)
    val testModelMeanTracked = testVectorizer.setFillWithMean.setTrackNulls(true).fit(testDataPercent)
    val testDataTransformedMeanTracked = testModelMeanTracked.transform(testDataPercent)
    val transformedValuesMeanTracked = testDataTransformedMeanTracked.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMeanTracked.schema.fieldNames should contain theSameElementsAs
      Array("inAPer", "inBPer", "inCPer", testVectorizer.outputName)

    val expectedMeanTracked = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 0.0, 2.0, 0.0, 0.0, 1.0)),
      (null, 2.0, null, Vectors.dense(3.0, 1.0, 2.0, 0.0, 0.0, 1.0)),
      (2.0, null, null, Vectors.dense(2.0, 0.0, 2.0, 1.0, 0.0, 1.0))
    )

    transformedValuesMeanTracked.map(_.get(0)) shouldEqual expectedMeanTracked.map(_._1)
    transformedValuesMeanTracked.map(_.get(1)) shouldEqual expectedMeanTracked.map(_._2)
    transformedValuesMeanTracked.map(_.get(2)) shouldEqual expectedMeanTracked.map(_._3)
    transformedValuesMeanTracked.map(_.get(3)) shouldEqual expectedMeanTracked.map(_._4)

    val fieldMetadata = testDataTransformedMeanTracked
      .select(testVectorizer.outputName).schema.fields
      .map(_.metadata).head
    val expectedMeta = TestOpVectorMetadataBuilder(
      testVectorizer,
      inAPer -> List(RootCol, IndCol(Some(Transmogrifier.NullString))),
      inBPer -> List(RootCol, IndCol(Some(Transmogrifier.NullString))),
      inCPer -> List(RootCol, IndCol(Some(Transmogrifier.NullString)))
    )
    OpVectorMetadata(testVectorizer.outputName, fieldMetadata) shouldBe expectedMeta
  }

  it should "work on columns of type Currency" in {
    val testVectorizer = new RealVectorizer().setInput(inACur, inBCur, inCCur)
    val testModelConstant = testVectorizer.setFillWithConstant(4.2).setTrackNulls(false).fit(testDataCurrency)
    val testDataTransformedConstant = testModelConstant.transform(testDataCurrency)

    testDataTransformedConstant.schema.fieldNames should contain theSameElementsAs
      Array("inACur", "inBCur", "inCCur", testVectorizer.outputName)

    val transformedValuesConstant = testDataTransformedConstant.collect()
    val expectedZero = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 2.0, 4.2)),
      (null, 2.0, null, Vectors.dense(4.2, 2.0, 4.2)),
      (2.0, null, null, Vectors.dense(2.0, 4.2, 4.2))
    )

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
      Array("inAPer", "inBPer", "inCPer", testVectorizer.outputName)

    val transformedValuesConstant = testDataTransformedConstant.collect()
    val expectedZero = Array(
      (4.0, 2.0, null, Vectors.dense(4.0, 2.0, 4.2)),
      (null, 2.0, null, Vectors.dense(4.2, 2.0, 4.2)),
      (2.0, null, null, Vectors.dense(2.0, 4.2, 4.2))
    )

    transformedValuesConstant.map(_.get(0)) shouldEqual expectedZero.map(_._1)
    transformedValuesConstant.map(_.get(1)) shouldEqual expectedZero.map(_._2)
    transformedValuesConstant.map(_.get(2)) shouldEqual expectedZero.map(_._3)
    transformedValuesConstant.map(_.get(3)) shouldEqual expectedZero.map(_._4)
  }
}
