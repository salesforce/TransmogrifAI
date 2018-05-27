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
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class GeolocationVectorizerTest extends FlatSpec with TestSparkContext {

  val (testData, inA, inB, inC, inD) = TestFeatureBuilder("inA", "inB", "inC", "inD",
    Seq[(Geolocation, Geolocation, Geolocation, Geolocation)](
      (Geolocation((32.4, -100.2, 3.0)), Geolocation((38.6, -110.4, 2.0)), Geolocation((39.1, -111.3, 3.0)),
        Geolocation.empty),
      (Geolocation((40.1, -120.3, 4.0)), Geolocation((42.5, -95.4, 4.0)), Geolocation.empty, Geolocation.empty),
      (Geolocation((45.0, -105.5, 4.0)), Geolocation.empty, Geolocation.empty, Geolocation.empty)
    )
  )
  private val testVectorizer = new GeolocationVectorizer().setInput(inA, inB, inC, inD)
  private val outputName = testVectorizer.operationName

  Spec[GeolocationVectorizer] should "have output name set correctly" in {
    testVectorizer.operationName shouldBe outputName
  }

  it should "throw an error if you try to get the output without setting the inputs" in {
    intercept[java.util.NoSuchElementException](new GeolocationVectorizer().getOutput())
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
    val testModelConstant = testVectorizer.setFillWithConstant(Geolocation(50.0, 50.0, GeolocationAccuracy.Street))
      .setTrackNulls(false).fit(testData)

    testModelConstant.parent shouldBe testVectorizer
    testModelConstant.transformFn(Seq(Geolocation.empty, Geolocation.empty, Geolocation.empty,
      Geolocation.empty)) shouldEqual
      Vectors.dense(50.0, 50.0, 4.0, 50.0, 50.0, 4.0, 50.0, 50.0, 4.0, 50.0, 50.0, 4.0).toOPVector

    val testDataTransformedConstant = testModelConstant.transform(testData)
    val transformedValuesConstant = testDataTransformedConstant.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedConstant.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", "inD", testModelConstant.getOutputFeatureName)

    val expectedConstant = Array(
      (Array(32.4, -100.2, 3.0), Array(38.6, -110.4, 2.0), Array(39.1, -111.3, 3.0), null,
        Vectors.dense(32.4, -100.2, 3.0, 38.6, -110.4, 2.0, 39.1, -111.3, 3.0, 50.0, 50.0, 4.0)),
      (Array(40.1, -120.3, 4.0), Array(42.5, -95.4, 4.0), null, null,
        Vectors.dense(40.1, -120.3, 4.0, 42.5, -95.4, 4.0, 50.0, 50.0, 4.0, 50.0, 50.0, 4.0)),
      (Array(45.0, -105.5, 4.0), null, null, null,
        Vectors.dense(45.0, -105.5, 4.0, 50.0, 50.0, 4.0, 50.0, 50.0, 4.0, 50.0, 50.0, 4.0))
    )

    transformedValuesConstant.map(_.get(0)) shouldEqual expectedConstant.map(_._1)
    transformedValuesConstant.map(_.get(1)) shouldEqual expectedConstant.map(_._2)
    transformedValuesConstant.map(_.get(2)) shouldEqual expectedConstant.map(_._3)
    transformedValuesConstant.map(_.get(3)) shouldEqual expectedConstant.map(_._4)
    transformedValuesConstant.map(_.get(4)) shouldEqual expectedConstant.map(_._5)
  }

  it should "fit the model with fillWithMean and transform data correctly" in {
    // Geometric means calculated from http://www.geomidpoint.com/
    val eps = 0.01
    val mean0 = Array(37.78, -106.24, 3.0)
    val mean1 = Array(40.79, -103.12, 0.0)
    val mean2 = Array(39.1, -111.3, 3.0)
    val mean3 = Array(0.0, 0.0, 0.0)

    val testModelMean = testVectorizer.setFillWithMean().setTrackNulls(false).fit(testData)
    val testDataTransformedMean = testModelMean.transform(testData)
    val transformedValuesMean = testDataTransformedMean.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMean.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", "inD", testModelMean.getOutputFeatureName)

    val expectedMean = Array(
      (Array(32.4, -100.2, 3.0), Array(38.6, -110.4, 2.0), Array(39.1, -111.3, 3.0), null,
        Vectors.dense(Array(32.4, -100.2, 3.0, 38.6, -110.4, 2.0, 39.1, -111.3, 3.0) ++ mean3)),
      (Array(40.1, -120.3, 4.0), Array(42.5, -95.4, 4.0), null, null,
        Vectors.dense(Array(40.1, -120.3, 4.0, 42.5, -95.4, 4.0) ++ mean2 ++ mean3)),
      (Array(45.0, -105.5, 4.0), null, null, null,
        Vectors.dense(Array(45.0, -105.5, 4.0) ++ mean1 ++ mean2 ++ mean3))
    )

    transformedValuesMean.map(_.get(0)) shouldEqual expectedMean.map(_._1)
    transformedValuesMean.map(_.get(1)) shouldEqual expectedMean.map(_._2)
    transformedValuesMean.map(_.get(2)) shouldEqual expectedMean.map(_._3)
    transformedValuesMean.map(_.get(3)) shouldEqual expectedMean.map(_._4)
    // For the last column, they contain the filled mean, to make sure to compare the doubles using the tolerance, eps
    transformedValuesMean.map(_.get(4)).toSeq.zip(expectedMean.map(_._5).toSeq).map(f => {
      val a = f._1.asInstanceOf[DenseVector].toArray
      val b = f._2.asInstanceOf[DenseVector].toArray
      withClue(s"Expected ${b.mkString(" ")}, got ${a.mkString(" ")}") {
        a.zip(b).map(g => math.abs(g._1 - g._2) should be < eps)
      }
    })
  }

  it should "keep track of null values if wanted, using fillWithMean" in {
    // Geometric means calculated from http://www.geomidpoint.com/
    val eps = 0.01
    // Add 1.0 at the end since comparison is with setTrackNulls = true
    val mean0 = Array(37.78, -106.24, 3.0, 1.0)
    val mean1 = Array(40.79, -103.12, 0.0, 1.0)
    val mean2 = Array(39.1, -111.3, 3.0, 1.0)
    val mean3 = Array(0.0, 0.0, 0.0, 1.0)

    val testModelMean = testVectorizer.setFillWithMean().setTrackNulls(true).fit(testData)
    val testDataTransformedMean = testModelMean.transform(testData)
    val transformedValuesMean = testDataTransformedMean.collect()

    // This is string because of vector type being private to spark ml
    testDataTransformedMean.schema.fieldNames should contain theSameElementsAs
      Array("inA", "inB", "inC", "inD", testModelMean.getOutputFeatureName)

    val expectedMean = Array(
      (Array(32.4, -100.2, 3.0), Array(38.6, -110.4, 2.0), Array(39.1, -111.3, 3.0), null,
        Vectors.dense(Array(32.4, -100.2, 3.0, 0.0, 38.6, -110.4, 2.0, 0.0, 39.1, -111.3, 3.0, 0.0) ++ mean3)),
      (Array(40.1, -120.3, 4.0), Array(42.5, -95.4, 4.0), null, null,
        Vectors.dense(Array(40.1, -120.3, 4.0, 0.0, 42.5, -95.4, 4.0, 0.0) ++ mean2 ++ mean3)),
      (Array(45.0, -105.5, 4.0), null, null, null,
        Vectors.dense(Array(45.0, -105.5, 4.0, 0.0) ++ mean1 ++ mean2 ++ mean3))
    )

    transformedValuesMean.map(_.get(0)) shouldEqual expectedMean.map(_._1)
    transformedValuesMean.map(_.get(1)) shouldEqual expectedMean.map(_._2)
    transformedValuesMean.map(_.get(2)) shouldEqual expectedMean.map(_._3)
    transformedValuesMean.map(_.get(3)) shouldEqual expectedMean.map(_._4)
    // For the last column, they contain the filled mean, to make sure to compare the doubles using the tolerance, eps
    transformedValuesMean.map(_.get(4)).toSeq.zip(expectedMean.map(_._5).toSeq).foreach(
      f => {
        val as = f._1.asInstanceOf[DenseVector].toArray
        val bs = f._2.asInstanceOf[DenseVector].toArray
        as.length shouldBe bs.length

        for {
          i <- as.indices
          a = as(i)
          b = bs(i)
        } {
          withClue(s"at $i:\nExpected ${bs.mkString(" ")},\n got     ${as.mkString(" ")}\n") {
            math.abs(a - b) should be < eps
          }
        }
      }
    )
    succeed
  }

  it should "have a working shortcut function" in {
    val testModelMean = testVectorizer.setFillWithMean().setTrackNulls(true).fit(testData)
    val testDataTransformedMean = testModelMean.transform(testData)
    val transformedValuesMean = testDataTransformedMean.collect()

    // Now using the shortcut
    val res = inA.vectorize(fillWithMean = true, trackNulls = true, others = Array(inB, inC, inD))
    val actualOutput = res.originStage.asInstanceOf[GeolocationVectorizer]
      .fit(testData).transform(testData).collect()

    for {
      i <- transformedValuesMean.indices
      expected = transformedValuesMean(i)
      actual = actualOutput(i)
    } {
      val actualData = actual.toSeq.toArray
      val expectedData = expected.toSeq.toArray
      withClue(s"failed at $i:") {
        actual shouldBe expected
      }
    }
  }

}
