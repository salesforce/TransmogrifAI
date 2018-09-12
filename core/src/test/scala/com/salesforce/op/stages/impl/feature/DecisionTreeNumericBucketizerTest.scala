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
import com.salesforce.op.stages.base.binary.BinaryModel
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.testkit.{RandomBinary, RandomReal}
import com.salesforce.op.utils.numeric.Number
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.Metadata
import org.junit.runner.RunWith
import org.scalatest.Matchers
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DecisionTreeNumericBucketizerTest extends OpEstimatorSpec[OPVector,
  BinaryModel[RealNN, Real, OPVector], DecisionTreeNumericBucketizer[Double, Real]]
  with DecisionTreeNumericBucketizerAsserts with AttributeAsserts
{
  val (inputData, estimator) = {
    val numericData = Seq(1.0.toReal, 18.0.toReal, Real.empty, (-1.23).toReal, 0.0.toReal)
    val labelData = Seq(1.0, 1.0, 0.0, 0.0, 1.0).map(_.toRealNN)
    val (inputData, numeric, label) = TestFeatureBuilder[Real, RealNN](numericData zip labelData)

    inputData -> new DecisionTreeNumericBucketizer[Double, Real]().setInput(label, numeric)
  }

  val expectedResult = Seq(
    Vectors.sparse(3, Array(1), Array(1.0)),
    Vectors.sparse(3, Array(1), Array(1.0)),
    Vectors.sparse(3, Array(2), Array(1.0)),
    Vectors.sparse(3, Array(0), Array(1.0)),
    Vectors.sparse(3, Array(1), Array(1.0))
  ).map(_.toOPVector)

  trait NormalData {
    val numericData: Seq[Real] = RandomReal.normal[Real]().withProbabilityOfEmpty(0.2).limit(1000)
    val labelData: Seq[RealNN] = RandomBinary(probabilityOfSuccess = 0.4).limit(1000).map(_.toDouble.toRealNN(0.0))
    val (ds, numeric, label) = TestFeatureBuilder[Real, RealNN](numericData zip labelData)
    val expectedSplits = Array.empty[Double]
    lazy val modelLocation = tempDir + "/dt-buck-test-model-" + org.joda.time.DateTime.now().getMillis
  }

  trait EmptyData {
    val (_, numeric, label) = TestFeatureBuilder[Real, RealNN](Seq[Real]() zip Seq[RealNN]())
    val nulls = Seq[(java.lang.Double, java.lang.Double)]((0.0: java.lang.Double) -> (null: java.lang.Double))
    import spark.implicits._
    val nullsDS = spark.createDataset(nulls).toDF(label.name, numeric.name)
    val emptyDS = spark.createDataset(Seq[(java.lang.Double, java.lang.Double)]()).toDF(label.name, numeric.name)
    val expectedSplits = Array.empty[Double]
  }

  trait UniformData {
    val (min, max) = (0.0, 100.0)
    val currencyData: Seq[Currency] =
      RandomReal.uniform[Currency](minValue = min, maxValue = max).withProbabilityOfEmpty(0.1).limit(1000)
    val labelData = currencyData.map(c => {
      c.value.map {
        case v if v < 15 => 0.0
        case v if v < 26 => 1.0
        case v if v < 91 => 2.0
        case _ => 3.0
      }.toRealNN(0.0)
    })
    val (ds, currency, label) = TestFeatureBuilder("currency", "label", currencyData zip labelData)
    val expectedSplits = Array(Double.NegativeInfinity, 15, 26, 91, Double.PositiveInfinity)
  }

  it should "produce output that is never a response, except the case where both inputs are" in new NormalData {
    Seq(
      label.copy(isResponse = false) -> numeric.copy(isResponse = false),
      label.copy(isResponse = true) -> numeric.copy(isResponse = false),
      label.copy(isResponse = false) -> numeric.copy(isResponse = true)
    ).foreach(inputs =>
      new DecisionTreeNumericBucketizer[Double, Real]().setInput(inputs).getOutput().isResponse shouldBe false
    )
    Seq(
      label.copy(isResponse = true) -> numeric.copy(isResponse = true)
    ).foreach(inputs =>
      new DecisionTreeNumericBucketizer[Double, Real]().setInput(inputs).getOutput().isResponse shouldBe true
    )
  }

  it should "not find any splits on random data" in new NormalData {
    val bucketizer = new DecisionTreeNumericBucketizer[Double, Real]().setInput(label, numeric).setTrackNulls(false)
    assertBucketizer(
      bucketizer, data = ds, shouldSplit = false, trackNulls = false, trackInvalid = false,
      expectedSplits = Array.empty, expectedTolerance = 0.0
    )
  }

  it should "work correctly on empty data" in new EmptyData {
    val bucketizer = new DecisionTreeNumericBucketizer[Double, Real]().setInput(label, numeric).setTrackNulls(false)
    assertBucketizer(
      bucketizer, data = nullsDS, shouldSplit = false, trackNulls = false, trackInvalid = false,
      expectedSplits = Array.empty, expectedTolerance = 0.0
    )
    a[IllegalArgumentException] should be thrownBy assertBucketizer(
      bucketizer, data = emptyDS, shouldSplit = false, trackNulls = false, trackInvalid = false,
      expectedSplits = Array.empty, expectedTolerance = 0.0
    )
  }

  it should "work as a shortcut" in new NormalData {
    val out = numeric.autoBucketize(label, trackNulls = true)
    out.originStage shouldBe a[DecisionTreeNumericBucketizer[_, _]]
    assertBucketizer(
      bucketizer = out.originStage.asInstanceOf[DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]]],
      data = ds, shouldSplit = false, trackNulls = true, trackInvalid = false,
      expectedSplits = Array.empty, expectedTolerance = 0.0
    )
  }

  it should "correctly bucketize when labels are specified" in new UniformData {
    val out = currency.autoBucketize(label = label, trackNulls = true, trackInvalid = true, minInfoGain = 0.1)
    assertBucketizer(
      bucketizer = out.originStage.asInstanceOf[DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]]],
      data = ds, shouldSplit = true, trackNulls = true, trackInvalid = true, expectedSplits = expectedSplits,
      expectedTolerance = 0.15
    )
  }

  it should "correctly bucketize over data with label leakage" in {
    val binaryData: Seq[Binary] = RandomBinary(probabilityOfSuccess = 0.5).withProbabilityOfEmpty(0.3).limit(1000)
    val currencyData: Seq[Currency] = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).limit(1000)
    val expectedRevenueData: Seq[Currency] = currencyData.zip(binaryData).map { case (cur, bi) =>
      bi.value match {
        case Some(v) => ((if (v) 1.0 else 0.0) * cur.value.get).toCurrency
        case None => Currency.empty
      }
    }
    val labelData = binaryData.map(_.toDouble.getOrElse(0.0).toRealNN)
    val rawData: Seq[(Binary, Currency, Currency, RealNN)] =
      binaryData.zip(currencyData).zip(expectedRevenueData).zip(labelData).map {
        case (((bi, cu), er), l) => (bi, cu, er, l)
      }

    val (ds, rawBinary, rawCurrency, rawER, label) =
      TestFeatureBuilder("binary", "currency", "expectedRevenue", "label", rawData)

    val out = rawER.autoBucketize(label.copy(isResponse = true), trackNulls = true, trackInvalid = true)
    assertBucketizer(
      bucketizer = out.originStage.asInstanceOf[DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]]],
      data = ds, shouldSplit = true, trackNulls = true, trackInvalid = true,
      expectedSplits = Array(Double.NegativeInfinity, 0.0, Double.PositiveInfinity),
      expectedTolerance = 0.15
    )
  }

  private def assertBucketizer
  (
    bucketizer: DecisionTreeNumericBucketizer[_, _ <: OPNumeric[_]],
    data: DataFrame,
    shouldSplit: Boolean,
    trackNulls: Boolean,
    trackInvalid: Boolean,
    expectedSplits: Array[Double],
    expectedTolerance: Double
  ): Unit = {
    val out = bucketizer.getOutput()
    val fitted = bucketizer.fit(data)
    fitted shouldBe a[DecisionTreeNumericBucketizerModel[_]]
    val model = fitted.asInstanceOf[DecisionTreeNumericBucketizerModel[_]]
    model.uid shouldBe bucketizer.uid
    model.operationName shouldBe bucketizer.operationName
    model.shouldSplit shouldBe shouldSplit
    model.trackNulls shouldBe trackNulls
    model.trackInvalid shouldBe trackInvalid

    val splits = model.splits
    assertSplits(splits = splits, expectedSplits = expectedSplits, expectedTolerance)

    val transformed = model.transform(data)
    val res = transformed.collect(out)
    val field = transformed.schema(out.name)
    assertNominal(field, Array.fill(res.head.value.size)(true), res)
      assertMetadata(
      shouldSplit = Array(shouldSplit),
      splits = Array(splits),
      trackNulls = trackNulls, trackInvalid = trackInvalid,
      model.getMetadata(), res
    )
  }


}

trait DecisionTreeNumericBucketizerAsserts {
  self: Matchers =>

  def assertSplits(splits: Array[Double], expectedSplits: Array[Double], expectedTolerance: Double): Unit = {
    withClue(
      s"Bucketizer splits: ${splits.mkString("[", ",", "]")}, expected: ${expectedSplits.mkString("[", ",", "]")}"
    ) {
      // account for an potential extra split
      (splits.length == expectedSplits.length || splits.length == expectedSplits.length + 1) shouldBe true
      // relative difference check per value
      val diff = splits.zip(expectedSplits).map { case (s, e) => math.abs((s - e) / math.max(s, e)) }
      diff.filter(Number.isValid).foreach(_ should be <= expectedTolerance)
    }
  }

  def assertMetadata(
    shouldSplit: Array[Boolean],
    splits: Array[Array[Double]],
    trackNulls: Boolean,
    trackInvalid: Boolean,
    stageMetadata: Metadata,
    res: Array[OPVector]
  ): Unit = {
    val numOfActualSplits = shouldSplit.count(_ == true)
    val expectedSize =
      splits.flatten.length - numOfActualSplits +
        (if (trackNulls) splits.length else 0) +
        (if (trackInvalid) numOfActualSplits else 0)

    for { v <- res } v.value.size shouldBe expectedSize

    val columnMeta: Array[OpVectorColumnMetadata] =
      stageMetadata.getMetadataArray(OpVectorMetadata.ColumnsKey)
        .flatMap(OpVectorColumnMetadata.fromMetadata)

    columnMeta.length shouldBe expectedSize
  }

}

