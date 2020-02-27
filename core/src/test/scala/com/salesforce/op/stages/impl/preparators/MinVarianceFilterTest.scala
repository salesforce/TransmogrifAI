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

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op._
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.MetadataParam
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.{DataFrame, Row}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

case class UnlabeledTextRawData
(
  id: String,
  textMap: Map[String, String]
)

@RunWith(classOf[JUnitRunner])
class MinVarianceFilterTest extends OpEstimatorSpec[OPVector, UnaryModel[OPVector, OPVector],
  UnaryEstimator[OPVector, OPVector]] {

  override def specName: String = Spec[MinVarianceFilter]

  private val textRawData = Seq(
    UnlabeledTextRawData("0", Map("color" -> "red", "fruit" -> "berry", "beverage" -> "tea")),
    UnlabeledTextRawData("1", Map("color" -> "orange", "fruit" -> "berry", "beverage" -> "coffee")),
    UnlabeledTextRawData("2", Map("color" -> "yello", "fruit" -> "berry", "beverage" -> "water")),
    UnlabeledTextRawData("3", Map("color" -> "green", "fruit" -> "berry")),
    UnlabeledTextRawData("4", Map("color" -> "blue", "fruit" -> "berry")),
    UnlabeledTextRawData("5", Map("color" -> "indigo", "fruit" -> "berry")),
    UnlabeledTextRawData("6", Map("fruit" -> "peach")),
    UnlabeledTextRawData("7", Map("fruit" -> "peach")),
    UnlabeledTextRawData("8", Map("fruit" -> "mango")),
    UnlabeledTextRawData("9", Map("beverage" -> "tea")),
    UnlabeledTextRawData("10", Map("beverage" -> "coffee")),
    UnlabeledTextRawData("11", Map("beverage" -> "water"))
  ).map(textRawData => (
    textRawData.id.toText,
    textRawData.textMap.toTextMap
  ))

  val (textData, id, textMap) = TestFeatureBuilder("id", "textMap", textRawData)
  // scalastyle:off
  private val data = Seq(
    SanityCheckDataTest("alex",     32,  5.0,  0,  1,  0.5,  0),
    SanityCheckDataTest("alice",    32,  4.0,  1,  0,  0,  0.1),
    SanityCheckDataTest("bob",      32,  6.0,  1,  1,  0.5,  0),
    SanityCheckDataTest("charles",  32,  5.5,  0,  1,  0.5,  0),
    SanityCheckDataTest("diana",    32,  5.4,  1,  0,  0,  0.1),
    SanityCheckDataTest("max",      32,  5.4,  1,  0,  0,  0.1)
  ).map(data =>
    Seq(data.age, data.height, data.height_null, data.gender, data.testFeatNegCor).toOPVector
  )
  // scalastyle:on

  val (testDataNoMeta, featureVector) = TestFeatureBuilder("features", data)

  val testMetadata = OpVectorMetadata(
    featureVector.name,
    Array("age", "height", "height_null", "gender", "testFeatNegCor").map { name =>
      OpVectorColumnMetadata(
        parentFeatureName = Seq(name),
        parentFeatureType = Seq(FeatureTypeDefaults.Real.getClass.getName),
        grouping = None,
        indicatorValue = None
      )
    },
    Seq("age", "height", "height_null", "gender", "testFeatNegCor")
      .map(name => name -> FeatureHistory(originFeatures = Seq(name), stages = Seq())).toMap
  )

  val featureNames = testMetadata.columns.map(_.makeColName())
  val expectedNamesFeatsDropped = Seq(featureNames(0), featureNames(3), featureNames(4))
  val expectedNamesFeatsKept = Seq(featureNames(1), featureNames(2))

  val testData = testDataNoMeta.select(
    testDataNoMeta(featureVector.name).as(featureVector.name, testMetadata.toMetadata)
  )

  val inputData = testData
  val estimator = new MinVarianceFilter().setRemoveBadFeatures(false).setInput(featureVector)
  val expectedResult = testData.select(featureVector.name).collect()
    .map(_.getAs[Vector](0).toOPVector).toSeq

  Spec[MinVarianceFilter] should "remove features" in {
    val checked = featureVector.minVariance(minVariance = 0.1, removeBadFeatures = true)
    val outputColName = checked.name

    checked.originStage shouldBe a[MinVarianceFilter]

    val model = checked.originStage.asInstanceOf[MinVarianceFilter].fit(testData)
    validateEstimatorOutput(outputColName, model, expectedNamesFeatsDropped)

    val transformedData = model.transform(testData)
    validateTransformerOutput(outputColName, transformedData, expectedNamesFeatsKept,
      expectedNamesFeatsDropped)
  }

  it should "have proper default values for all params" in {
    val filter = new MinVarianceFilter()
    filter.getOrDefault(filter.minVariance) shouldBe MinVarianceFilter.MinVariance
    filter.getOrDefault(filter.removeBadFeatures) shouldBe MinVarianceFilter.RemoveBadFeatures
  }

  it should "consume the output from an OP vectorizer and remove trouble features accordingly" in {
    val minVarianceFilter = new MinVarianceFilter()
    val outputColName = minVarianceFilter
      .setMinVariance(0.1)
      .setRemoveBadFeatures(true)
      .setInput(featureVector).getOutput().name

    val model = minVarianceFilter.fit(testData)
    validateEstimatorOutput(outputColName, model, expectedNamesFeatsDropped)

    val transformedData = model.transform(testData)
    val metadata: Metadata = getMetadata(outputColName, transformedData)
    val summary = MinVarianceSummary.fromMetadata(metadata.getSummaryMetadata())
    val outputColumns = OpVectorMetadata(transformedData.schema(outputColName))
    testMetadata.columns.map(_.makeColName()) should contain theSameElementsAs summary.names
    outputColumns.columns.length + summary.dropped.length shouldEqual testMetadata.columns.length

  }

  it should "not remove trouble features" in {
    val minVarianceFilter = new MinVarianceFilter()
    val outputColName = minVarianceFilter
      .setMinVariance(0.1)
      .setRemoveBadFeatures(false)
      .setInput(featureVector).getOutput().name
    val model = minVarianceFilter.fit(testData)
    validateEstimatorOutput(outputColName, model, Seq.empty)

    val transformedData = model.transform(testData)
    validateTransformerOutput(outputColName, transformedData, featureNames, Seq.empty)
  }

  it should "fail when input dataset is empty" in {
    val emptyData = Seq.empty[SanityCheckDataTest].map(data =>
      Seq(data.age, data.height, data.height_null, data.gender, data.testFeatNegCor).toOPVector
    )

    val (emptyDataNoMeta, emptyFeatureVector) = TestFeatureBuilder("features", emptyData)
    val minVarianceFilter = new MinVarianceFilter()
      .setMinVariance(0.1)
      .setRemoveBadFeatures(true)
      .setInput(emptyFeatureVector)

    val emptyTestData = emptyDataNoMeta.select(
      emptyDataNoMeta(emptyFeatureVector.name).as(emptyFeatureVector.name, testMetadata.toMetadata)
    )

    the[IllegalArgumentException] thrownBy {
      minVarianceFilter.fit(emptyTestData)
    } should have message "requirement failed: Sample size cannot be zero"
  }

  it should "not be robust to missing metadata or partial metadata on inputs" in {
    val (data, f1) = TestFeatureBuilder[OPVector]("features",
      Seq(Vectors.dense(5.0, 1, 1, 0)).map(_.toOPVector)
    )

    val f1c = f1.copy()
    val minVarianceFilter = new MinVarianceFilter().setInput(f1c)
    assertThrows[IllegalArgumentException] {
      minVarianceFilter.fit(data).transform(data)
    }
  }

  it should "throw an error if all of the features are removed" in {
    val minVarianceFilter = new MinVarianceFilter()
      .setMinVariance(1000)
      .setRemoveBadFeatures(true)
      .setInput(featureVector)

    the[RuntimeException] thrownBy {
      minVarianceFilter.fit(testData)
    } should have message
      "requirement failed: The minimum variance filter has dropped all of your features, " +
        "check your input data or your threshold"
  }

  it should "not fail when maps have the same keys" in {
    val mapData = textRawData.map {
      case (i, tm) => (i, tm.value.toPickListMap, tm.value.toPickListMap,
        tm.value.map { case (k, _) => k -> math.random }.toRealMap)
    }
    val (mapDataFrame, id, plMap1, plMap2, doubleMap) = TestFeatureBuilder(
      "id", "textMap1", "textMap2", "doubleMap", mapData)
    val features = Seq(id, plMap1, plMap2, doubleMap).transmogrify()
    val filtered = features.minVariance()
    val output = new OpWorkflow().setResultFeatures(filtered).transform(mapDataFrame)
    output.select(filtered.name).count() shouldBe 12

    val metadata = filtered.originStage.getMetadata()
    val summary = MinVarianceSummary.fromMetadata(metadata.getSummaryMetadata())
    summary.dropped.size shouldBe 0
  }

  it should "produce the same statistics if the same transformation is applied twice" in {
    val plMap = textMap.map[PickListMap](_.value.toPickListMap)
    val features = Seq(id, plMap, plMap).transmogrify()
    val filtered = features.minVariance()
    val output = new OpWorkflow().setResultFeatures(filtered).transform(textData)
    output.select(filtered.name).count() shouldBe 12

    val metadata = filtered.originStage.getMetadata()
    val summary = MinVarianceSummary.fromMetadata(metadata.getSummaryMetadata())
    summary.dropped.size shouldBe 0
  }

  private def validateEstimatorOutput
  (
    outputColName: String,
    model: UnaryModel[OPVector, OPVector],
    expectedFeaturesToDrop: Seq[String]
  ): Unit = {
    val metadata = model.getMetadata()
    val summary = MinVarianceSummary.fromMetadata(metadata.getSummaryMetadata())
    val featuresKept = OpVectorMetadata(outputColName, metadata).columns.length

    summary.dropped should contain theSameElementsAs expectedFeaturesToDrop
    featuresKept equals summary.names.toSet.diff(summary.dropped.toSet).size
    summary.featuresStatistics.variance.min shouldBe 0.0
  }

  private def validateTransformerOutput
  (
    outputColName: String,
    transformedData: DataFrame,
    expectedFeatNames: Seq[String],
    expectedFeaturesToDrop: Seq[String]
  ): Unit = {
    val expectedLength = expectedFeatNames.length - expectedFeaturesToDrop.length
    transformedData.select(outputColName).collect().foreach { case Row(features: Vector) =>
      features.toArray.length equals expectedLength
    }
  }

  private def getMetadata(outputColName: String, transformedData: DataFrame): Metadata = {
    val transformedDataSchema = transformedData.schema
    val metadataOrig = transformedDataSchema(outputColName).metadata
    val metadata = validateEncodeDecode(metadataOrig)
    metadata
  }

  private def validateEncodeDecode(metadata: Metadata): Metadata = {
    val metaParam = new MetadataParam("parent", "name", "doc")
    val jsonEncodedMeta = metaParam.jsonEncode(metadata)
    val jsonDecodedMeta = metaParam.jsonDecode(jsonEncodedMeta)
    jsonDecodedMeta.hashCode() shouldEqual metadata.hashCode()
    jsonDecodedMeta
  }
}
