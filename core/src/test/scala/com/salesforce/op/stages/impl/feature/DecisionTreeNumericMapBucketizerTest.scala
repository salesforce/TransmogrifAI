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

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryModel
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.testkit.{RandomBinary, RandomReal}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner



@RunWith(classOf[JUnitRunner])
class DecisionTreeNumericMapBucketizerTest extends OpEstimatorSpec[OPVector,
  BinaryModel[RealNN, RealMap, OPVector], DecisionTreeNumericMapBucketizer[Double, RealMap]]
  with DecisionTreeNumericBucketizerAsserts
{
  import OPMapVectorizerTestHelper._

  val (inputData, estimator) = {
    val numericData = Seq(
      Map("a" -> 1.0),
      Map("a" -> 18.0),
      Map("b" -> 0.0),
      Map("a" -> -1.23, "b" -> 1.0),
      Map("a" -> -1.23, "b" -> 1.0, "c" -> 117.0)
    ).map(_.toRealMap)
    val labelData = Seq(1.0, 1.0, 0.0, 0.0, 1.0).map(_.toRealNN)
    val (inputData, numeric, label) = TestFeatureBuilder[RealMap, RealNN](numericData zip labelData)

    inputData -> new DecisionTreeNumericMapBucketizer[Double, RealMap]().setInput(label, numeric)
  }

  val expectedResult = Seq(
    Vectors.sparse(7, Array(1, 5, 6), Array(1.0, 1.0, 1.0)),
    Vectors.sparse(7, Array(1, 5, 6), Array(1.0, 1.0, 1.0)),
    Vectors.sparse(7, Array(2, 3, 6), Array(1.0, 1.0, 1.0)),
    Vectors.sparse(7, Array(0, 4, 6), Array(1.0, 1.0, 1.0)),
    Vectors.sparse(7, Array(0, 4), Array(1.0, 1.0))
  ).map(_.toOPVector)

  trait NormalData {
    val total = 1000
    val numerics = RandomReal.normal[Real]().withProbabilityOfEmpty(0.2)
    val labelData: Seq[RealNN] = RandomBinary(probabilityOfSuccess = 0.4).limit(total).map(_.toDouble.toRealNN(0.0))
    val rawData =
      numerics.limit(total).zip(numerics.limit(total)).zip(numerics.limit(total)).zip(labelData)
        .map { case (((f1, f2), f3), f4) => (f1, f2, f3, f4) }
    val (data, f1, f2, f3, label) = TestFeatureBuilder[Real, Real, Real, RealNN](rawData)
    val realMapFeature = makeTernaryOPMapTransformer[Real, RealMap, Double](f1, f2, f3)
    lazy val modelLocation = tempDir + "/dt-map-buck-test-model-" + org.joda.time.DateTime.now().getMillis
  }

  trait UniformData {
    val total = 1000
    val (min, max) = (0.0, 100.0)
    val currencies: RandomReal[Currency] =
      RandomReal.uniform[Currency](minValue = min, maxValue = max).withProbabilityOfEmpty(0.1)
    val correlated = currencies.limit(total)
    val labelData = correlated.map(c => {
      c.value.map {
        case v if v < 15 => 0.0
        case v if v < 26 => 1.0
        case v if v < 91 => 2.0
        case _ => 3.0
      }.toRealNN(0.0)
    })
    val rawData =
      correlated.zip(currencies.limit(total)).zip(correlated).zip(labelData)
        .map { case (((f1, f2), f3), f4) => (f1, f2, f3, f4) }
    val (data, f1, f2, f3, label) = TestFeatureBuilder[Currency, Currency, Currency, RealNN](rawData)
    val currencyMapFeature = makeTernaryOPMapTransformer[Currency, CurrencyMap, Double](f1, f2, f3)
    val expectedSplits = Array(Double.NegativeInfinity, 15, 26, 91, Double.PositiveInfinity)
  }

  lazy val (data, target, currencyMap, realMap) = TestFeatureBuilder("target", "currencyMap", "realMap2",
    Seq[(RealNN, CurrencyMap, RealMap)](
      (1.0.toRealNN, CurrencyMap(Map("c0" -> 10)), RealMap.empty),
      (1.0.toRealNN, CurrencyMap(Map("c0" -> 10)), RealMap.empty),
      (1.0.toRealNN, CurrencyMap(Map("c0" -> 8)), RealMap.empty),
      (0.0.toRealNN, CurrencyMap(Map("c0" -> 5)), RealMap.empty),
      (0.0.toRealNN, CurrencyMap(Map("c0" -> 3)), RealMap.empty),
      (0.0.toRealNN, CurrencyMap(Map("c0" -> 0)), RealMap.empty)
    )
  )

  it should "produce output that is never a response, except the case where both inputs are" in new NormalData {
    Seq(
      label.copy(isResponse = false) -> realMapFeature.copy(isResponse = false),
      label.copy(isResponse = true) -> realMapFeature.copy(isResponse = false),
      label.copy(isResponse = false) -> realMapFeature.copy(isResponse = true)
    ).foreach(inputs =>
      new DecisionTreeNumericMapBucketizer[Double, RealMap]().setInput(inputs).getOutput().isResponse shouldBe false
    )
    Seq(
      label.copy(isResponse = true) -> realMapFeature.copy(isResponse = true)
    ).foreach(inputs =>
      new DecisionTreeNumericMapBucketizer[Double, RealMap]().setInput(inputs).getOutput().isResponse shouldBe true
    )
  }

  it should "serialize correctly" in new NormalData {
    val bucketizer = new DecisionTreeNumericMapBucketizer[Double, RealMap]()
      .setInput(label, realMapFeature).setTrackNulls(false)
    val workflow = new OpWorkflow()
    val model = workflow.setInputDataset(data).setResultFeatures(bucketizer.getOutput()).train()
    model.save(modelLocation)
    val loaded = workflow.loadModel(modelLocation)
    loaded.stages.map(_.uid) should contain (bucketizer.uid)
  }

  it should "not find any splits on random data" in new NormalData {
    val bucketizer = new DecisionTreeNumericMapBucketizer[Double, RealMap]()
      .setInput(label, realMapFeature).setTrackNulls(false)

    assertBucketizer(bucketizer, data,
      shouldSplitByKey = Map(f1.name -> false, f2.name -> false, f3.name -> false),
      splitsByKey = Map(f1.name -> Array(), f2.name -> Array(), f3.name -> Array()),
      trackNulls = false, trackInvalid = false,
      expectedTolerance = 0.0
    )
  }

  it should "work as a shortcut" in new NormalData {
    val out = realMapFeature.autoBucketize(label, trackNulls = true)
    out.originStage shouldBe a[DecisionTreeNumericMapBucketizer[_, _]]

    assertBucketizer(out.originStage.asInstanceOf[DecisionTreeNumericMapBucketizer[_, _ <: OPMap[_]]], data,
      shouldSplitByKey = Map(f1.name -> false, f2.name -> false, f3.name -> false),
      splitsByKey = Map(f1.name -> Array(), f2.name -> Array(), f3.name -> Array()),
      trackNulls = true, trackInvalid = false,
      expectedTolerance = 0.0
    )
  }

  it should "correctly bucketize when labels are specified" in new UniformData {
    val out = currencyMapFeature.autoBucketize(label = label, trackNulls = true, trackInvalid = true, minInfoGain = 0.1)
    out.originStage shouldBe a[DecisionTreeNumericMapBucketizer[_, _]]

    assertBucketizer(out.originStage.asInstanceOf[DecisionTreeNumericMapBucketizer[_, _ <: OPMap[_]]], data,
      shouldSplitByKey = Map(f1.name -> true, f2.name -> false, f3.name -> true),
      splitsByKey = Map(f1.name -> expectedSplits, f2.name -> Array(), f3.name -> expectedSplits),
      trackNulls = true, trackInvalid = true,
      expectedTolerance = 0.15
    )
  }

  it should "drop empty numeric map" in {
    val targetResponse = target.copy(isResponse = true)
    val currencyMapBkts = currencyMap.autoBucketize(label = targetResponse, trackNulls = false, minInfoGain = 0.1)
    val realMapBkts = realMap.autoBucketize(label = targetResponse, trackNulls = false, minInfoGain = 0.1)
    val featureVector = Seq(currencyMapBkts, realMapBkts).transmogrify(Some(targetResponse))

    val transformed = new OpWorkflow().setResultFeatures(currencyMapBkts, realMapBkts, featureVector).transform(data)

    // featureVector should consist of bucketized features from currencyMap and no feature from realMap
    val featureVectorMeta = OpVectorMetadata(transformed.schema(featureVector.name))
    featureVectorMeta.columns.length shouldBe 2
    featureVectorMeta.columns.foreach{ col =>
      col.parentFeatureName should contain theSameElementsAs Seq("currencyMap")
      col.parentFeatureType should contain theSameElementsAs Seq("com.salesforce.op.features.types.CurrencyMap")
      col.grouping shouldBe Some("c0")
    }
  }

  private def assertBucketizer
  (
    bucketizer: DecisionTreeNumericMapBucketizer[_, _ <: OPMap[_]],
    data: DataFrame,
    shouldSplitByKey: Map[String, Boolean],
    splitsByKey: Map[String, Array[Double]],
    trackNulls: Boolean,
    trackInvalid: Boolean,
    expectedTolerance: Double
  ): Unit = {
    val out = bucketizer.getOutput()
    val model = new OpWorkflow().setResultFeatures(out).setInputDataset(data).train()
    val fitted = model.getOriginStageOf(out)

    fitted shouldBe a[DecisionTreeNumericMapBucketizerModel[_]]
    val stage = fitted.asInstanceOf[DecisionTreeNumericMapBucketizerModel[_]]
    stage.uid shouldBe bucketizer.uid
    stage.operationName shouldBe bucketizer.operationName
    stage.shouldSplitByKey.keys should contain theSameElementsAs stage.splitsByKey.keys
    stage.shouldSplitByKey shouldBe shouldSplitByKey
    stage.trackNulls shouldBe trackNulls
    stage.trackInvalid shouldBe trackInvalid

    stage.splitsByKey.size shouldBe splitsByKey.size
    stage.splitsByKey.keys should contain theSameElementsAs splitsByKey.keys

    stage.splitsByKey.keys.foreach(k =>
      assertSplits(splits = stage.splitsByKey(k), expectedSplits = splitsByKey(k), expectedTolerance)
    )
    val scored = model.setInputDataset(data).score(keepIntermediateFeatures = true)
    val res = scored.collect(out)
    val field = scored.schema(out.name)
    AttributeTestUtils.assertNominal(field, Array.fill(res.head.value.size)(true))
    assertMetadata(
      shouldSplit = stage.shouldSplitByKey.toArray.sortBy(_._1).map(_._2),
      splits = stage.splitsByKey.toArray.sortBy(_._1).map(_._2),
      trackNulls = trackNulls, trackInvalid = trackInvalid,
      stage.getMetadata(), res
    )

  }

}
