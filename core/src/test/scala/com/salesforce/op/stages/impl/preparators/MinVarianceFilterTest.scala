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
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.MetadataParam
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.feature.RealNNVectorizer
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.{DataFrame, Row}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

case class MinVarianceFilterDataTest
(
  name: String,
  age: Double,
  height: Double,
  height_null: Double,
  isBlueEyed: Double,
  gender: Double,
  testFeatNegCor: Double
)

case class SCDataTest(label: RealNN, features: OPVector)

case class TextRawData
(
  id: String,
  target: Double,
  textMap: Map[String, String]
)

@RunWith(classOf[JUnitRunner])
class MinVarianceCheckerTest extends OpEstimatorSpec[OPVector, BinaryModel[RealNN, OPVector, OPVector],
  BinaryEstimator[RealNN, OPVector, OPVector]] {

  override def specName: String = Spec[MinVarianceFilter]

  // loggingLevel(Level.INFO)

  private val textRawData = Seq(
    TextRawData("0", 1.0, Map("color" -> "red", "fruit" -> "berry", "beverage" -> "tea")),
    TextRawData("1", 1.0, Map("color" -> "orange", "fruit" -> "berry", "beverage" -> "coffee")),
    TextRawData("2", 1.0, Map("color" -> "yello", "fruit" -> "berry", "beverage" -> "water")),
    TextRawData("3", 1.0, Map("color" -> "green", "fruit" -> "berry")),
    TextRawData("4", 1.0, Map("color" -> "blue", "fruit" -> "berry")),
    TextRawData("5", 1.0, Map("color" -> "indigo", "fruit" -> "berry")),
    TextRawData("6", 0.0, Map("fruit" -> "peach")),
    TextRawData("7", 0.0, Map("fruit" -> "peach")),
    TextRawData("8", 0.0, Map("fruit" -> "mango")),
    TextRawData("9", 0.0, Map("beverage" -> "tea")),
    TextRawData("10", 0.0, Map("beverage" -> "coffee")),
    TextRawData("11", 0.0, Map("beverage" -> "water"))
  ).map( textRawData => (
    textRawData.id.toText,
    textRawData.target.toRealNN,
    textRawData.textMap.toTextMap
  ))

  val (textData, id, target, textMap) = TestFeatureBuilder("id", "target", "textMap", textRawData)
  val targetResponse: FeatureLike[RealNN] = target.copy(isResponse = true)

  // scalastyle:off
  private val data = Seq(
    SanityCheckDataTest("alex",     32,  5.0,  0,  1,  0.5,  0),
    SanityCheckDataTest("alice",    32,  4.0,  1,  0,  0,  0.1),
    SanityCheckDataTest("bob",      32,  6.0,  1,  1,  0.5,  0),
    SanityCheckDataTest("charles",  32,  5.5,  0,  1,  0.5,  0),
    SanityCheckDataTest("diana",    32,  5.4,  1,  0,  0,  0.1),
    SanityCheckDataTest("max",      32,  5.4,  1,  0,  0,  0.1)
  ).map(data =>
    (
      data.isBlueEyed.toRealNN,
      Seq(data.age, data.height, data.height_null, data.gender, data.testFeatNegCor).toOPVector
    )
  )
  // scalastyle:on

  val (testDataNoMeta, targetLabelNoResponse, featureVector) = TestFeatureBuilder("isBlueEye", "features", data)

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
      .map( name => name -> FeatureHistory(originFeatures = Seq(name), stages = Seq())).toMap
  )

  val featureNames = testMetadata.columns.map(_.makeColName())
  val featuresToDrop = Seq(featureNames(0), featureNames(3), featureNames(4))
  val expectedCorrFeatNames = featureNames.tail
  val expectedCorrFeatNamesIsNan = Seq(featureNames(0))

  val testData = testDataNoMeta.select(
    testDataNoMeta(targetLabelNoResponse.name),
    testDataNoMeta(featureVector.name).as(featureVector.name, testMetadata.toMetadata)
  )

  val targetLabel = targetLabelNoResponse.copy(isResponse = true)

  val inputData = testData
  val estimator = new MinVarianceChecker().setRemoveBadFeatures(false).setInput(targetLabel, featureVector)
  val expectedResult = testData.select(featureVector.name).collect()
    .map(_.getAs[Vector](0).toOPVector).toSeq

  Spec[MinVarianceChecker] should "remove trouble features" in {
    val checked = targetLabel.sanityCheck(featureVector,
      maxCorrelation = 0.99, minVariance = 0.0, checkSample = 1.0, removeBadFeatures = true)
    val outputColName = checked.name

    checked.originStage shouldBe a[MinVarianceChecker]

    val model = checked.originStage.asInstanceOf[MinVarianceChecker].fit(testData)
    val featuresToDrop = Seq(featureNames(0), featureNames(3), featureNames(4))
    validateEstimatorOutput(outputColName, model, featuresToDrop, targetLabel.name)

    val transformedData = model.transform(testData)
    val expectedFeatNames = expectedCorrFeatNames ++ expectedCorrFeatNamesIsNan
    validateTransformerOutput(outputColName, transformedData, expectedFeatNames, expectedCorrFeatNames,
      featuresToDrop, expectedCorrFeatNamesIsNan)
  }

  it should "not allow setting invalid param values" in {
    val checker = new MinVarianceChecker()
    the[IllegalArgumentException] thrownBy checker.setCheckSample(-1.0)
    the[IllegalArgumentException] thrownBy checker.setCheckSample(0.0)
    the[IllegalArgumentException] thrownBy checker.setCheckSample(2.0)
    the[IllegalArgumentException] thrownBy checker.setMinCorrelation(-1.0)
    the[IllegalArgumentException] thrownBy checker.setMinCorrelation(2.0)
    the[IllegalArgumentException] thrownBy checker.setMaxCorrelation(-1.0)
    the[IllegalArgumentException] thrownBy checker.setMaxCorrelation(2.0)
    the[IllegalArgumentException] thrownBy checker.setSampleUpperLimit(-1)
    the[IllegalArgumentException] thrownBy checker.setSampleLowerLimit(-1)
  }

  it should "have proper default values for all params" in {
    val checker = new MinVarianceChecker()
    checker.getOrDefault(checker.sampleLowerLimit) shouldBe MinVarianceChecker.SampleLowerLimit
    checker.getOrDefault(checker.sampleUpperLimit) shouldBe MinVarianceChecker.SampleUpperLimit
    checker.getOrDefault(checker.checkSample) shouldBe MinVarianceChecker.CheckSample
    checker.getOrDefault(checker.maxCorrelation) shouldBe MinVarianceChecker.MaxCorrelation
    checker.getOrDefault(checker.minVariance) shouldBe MinVarianceChecker.MinVariance
    checker.getOrDefault(checker.minCorrelation) shouldBe MinVarianceChecker.MinCorrelation
    checker.getOrDefault(checker.correlationType) shouldBe CorrelationType.Pearson.entryName
    checker.getOrDefault(checker.removeBadFeatures) shouldBe MinVarianceChecker.RemoveBadFeatures
  }

  it should "consume the output from an OP vectorizer and remove trouble features accordingly" in {
    val MinVarianceChecker = new MinVarianceChecker()

    val outputColName = MinVarianceChecker
      .setMaxCorrelation(.99)
      .setMinVariance(0.0)
      .setCheckSample(1.0)
      .setRemoveBadFeatures(true)
      .setInput(targetLabel, featureVector).getOutput().name

    val model = MinVarianceChecker.fit(testData)
    validateEstimatorOutput(outputColName, model, featuresToDrop, targetLabel.name)

    val transformedData = model.transform(testData)
    val metadata: Metadata = getMetadata(outputColName, transformedData)
    val summary = MinVarianceCheckerSummary.fromMetadata(metadata.getSummaryMetadata())
    val outputColumns = OpVectorMetadata(transformedData.schema(outputColName))
    targetLabel.name +: testMetadata.columns.map(_.makeColName()) should
      contain theSameElementsAs summary.names
    outputColumns.columns.length + summary.dropped.length shouldEqual testMetadata.columns.length

    val expectedFeatNames = expectedCorrFeatNames ++ expectedCorrFeatNamesIsNan
    validateTransformerOutput(outputColName, transformedData, expectedFeatNames, expectedCorrFeatNames,
      featuresToDrop, expectedCorrFeatNamesIsNan)
  }

  it should "not remove trouble features" in {
    val MinVarianceChecker = new MinVarianceChecker()

    val outputColName = MinVarianceChecker
      .setMaxCorrelation(.99)
      .setMinVariance(0.0)
      .setCheckSample(1.0)
      .setInput(targetLabel, featureVector).getOutput().name

    val model = MinVarianceChecker.fit(testData)
    validateEstimatorOutput(outputColName, model, Seq(), targetLabel.name)

    val transformedData = model.transform(testData)

    val expectedFeatNames = expectedCorrFeatNames ++ expectedCorrFeatNamesIsNan
    validateTransformerOutput(outputColName, transformedData, expectedFeatNames, expectedCorrFeatNames,
      Seq(), expectedCorrFeatNamesIsNan)
  }

  it should "fail when not enough data to run" in {
    val MinVarianceChecker = new MinVarianceChecker()

    val outputColName = MinVarianceChecker
      .setMaxCorrelation(.99)
      .setMinVariance(0.0)
      .setCheckSample(0.000001)
      .setRemoveBadFeatures(true)
      .setInput(targetLabel, featureVector).getOutput().name

    val result = MinVarianceChecker.fit(testData)
    succeed
  }

  it should "fail when features are defined incorrectly" in {
    the[IllegalArgumentException] thrownBy {
      new MinVarianceChecker().setInput(targetLabel.copy(isResponse = true), featureVector.copy(isResponse = true))
    } should have message "The feature vector should not contain any response features."
  }

  it should "not be robust to missing metadata or partial metadata on inputs" in {

    val (data, f1, f2) = TestFeatureBuilder[RealNN, OPVector]("label", "features",
      Seq((RealNN(32), Vectors.dense(5.0, 1, 1, 0).toOPVector),
        (RealNN(32), Vectors.dense(4.0, 0, 0, 1).toOPVector),
        (RealNN(34), Vectors.dense(6.0, 1, 1, 0).toOPVector),
        (RealNN(32), Vectors.dense(5.5, 1, 1, 0).toOPVector),
        (RealNN(30), Vectors.dense(5.4, 0, 0, 1).toOPVector),
        (RealNN(32), Vectors.dense(5.4, 0, 0, 1).toOPVector))
    )

    val f1r = f1.copy(isResponse = true)
    val MinVarianceChecker = new MinVarianceChecker()
      .setInput(f1r, f2)
      .setRemoveBadFeatures(true)
      .setCheckSample(1.0)
    val output = MinVarianceChecker.getOutput()
    assertThrows[IllegalArgumentException] {
      MinVarianceChecker.fit(data).transform(data)
    }
  }

  it should "correctly limit the sample size if requested to" in {
    val (data, f1, f2) = TestFeatureBuilder[RealNN, OPVector]("label", "features",
      Seq((RealNN(32), Vectors.dense(5.0, 1, 1, 0).toOPVector),
        (RealNN(32), Vectors.dense(4.0, 0, 0, 1).toOPVector),
        (RealNN(34), Vectors.dense(6.0, 1, 1, 0).toOPVector),
        (RealNN(32), Vectors.dense(5.5, 1, 1, 0).toOPVector),
        (RealNN(30), Vectors.dense(5.4, 0, 0, 1).toOPVector),
        (RealNN(32), Vectors.dense(5.4, 0, 0, 1).toOPVector))
    )

    val f1r = f1.copy(isResponse = true)
    // Check that setSampleUpperLimit works by limiting the sample size to 0 (deterministic)
    // and checking for an error.
    // We cannot use a non-zero sample limit since the Spark sampling algorithm
    // is not deterministic and does not guarantee exactly the supplied sampling fraction
    val MinVarianceChecker = new MinVarianceChecker()
      .setInput(f1r, f2)
      .setRemoveBadFeatures(true)
      .setCheckSample(0.99999)
      .setSampleLowerLimit(0)
      .setSampleUpperLimit(0)

    the[IllegalArgumentException] thrownBy {
      MinVarianceChecker.fit(data)
    } should have message "requirement failed: Sample size cannot be zero"
  }

  it should "compute higher spearman correlation for monotonic, nonlinear functions than pearson" in {
    val x = 1.0 to 20.0 by 1.0
    val xSquare = x.map(Math.pow(_, 5))
    val (data, labelNoResponse, feature) = TestFeatureBuilder[RealNN, RealNN]("label", "feature",
      x.map(_.toRealNN).zip(xSquare.map(_.toRealNN))
    )

    val label = labelNoResponse.copy(isResponse = true)

    val vectorized = feature.vectorize()
    val dataVectorized = vectorized.originStage.asInstanceOf[RealNNVectorizer].transform(data)

    def getFeatureCorrelationFor(MinVarianceChecker: MinVarianceChecker): Double = {
      val model = MinVarianceChecker.fit(dataVectorized)

      val summaryMeta = model.getMetadata().getSummaryMetadata()
      val correlations = summaryMeta
        .getMetadata(MinVarianceCheckerNames.CorrelationsWLabel)
        .getDoubleArray(MinVarianceCheckerNames.Values)
      correlations(0)
    }

    val MinVarianceCheckerSpearman = new MinVarianceChecker()
      .setInput(label, vectorized)
      .setCorrelationType(CorrelationType.Spearman)
      .setCheckSample(0.99999)

    val MinVarianceCheckerPearson = new MinVarianceChecker()
      .setInput(label, vectorized)
      .setCorrelationType(CorrelationType.Pearson)
      .setCheckSample(0.99999)

    assert(
      getFeatureCorrelationFor(MinVarianceCheckerSpearman) > getFeatureCorrelationFor(MinVarianceCheckerPearson)
    )
  }

  it should "throw an error if all of the features are removed" in {
    val MinVarianceChecker = new MinVarianceChecker()

    val outputColName = MinVarianceChecker
      .setMaxCorrelation(.000001)
      .setMinVariance(1000)
      .setCheckSample(0.999999)
      .setRemoveBadFeatures(true)
      .setInput(targetLabel, featureVector).getOutput().name

    the[RuntimeException] thrownBy {
      MinVarianceChecker.fit(testData)
    } should have message
      "requirement failed: The sanity checker has dropped all of your features, check your input data quality"
  }

  it should "not fail when maps have the same keys" in {
    val mapData = textRawData.map{
      case (i, t, tm) => (i, t, tm.value.toPickListMap, tm.value.toPickListMap,
        tm.value.map{ case (k, v) => k -> math.random }.toRealMap)
    }
    val (mapDataFrame, id, target, plMap1, plMap2, doubleMap) = TestFeatureBuilder(
      "id", "target", "textMap1", "textMap2", "doubleMap", mapData)
    val targetResponse: FeatureLike[RealNN] = target.copy(isResponse = true)
    val features = Seq(id, target, plMap1, plMap2, doubleMap).transmogrify()
    val checked = targetResponse.sanityCheck(features, categoricalLabel = Option(true))
    val output = new OpWorkflow().setResultFeatures(checked).transform(mapDataFrame)
    output.select(checked.name).count() shouldBe 12
    val meta = MinVarianceCheckerSummary.fromMetadata(checked.originStage.getMetadata().getSummaryMetadata())
    meta.dropped.size shouldBe 0
    meta.categoricalStats.size shouldBe 10
    meta.categoricalStats.foreach(_.contingencyMatrix("0").length shouldBe 2)
  }

  it should "produce the same statistics if the same transformation is applied twice" in {
    val plMap = textMap.map[PickListMap](_.value.toPickListMap)
    val features = Seq(id, target, plMap, plMap).transmogrify()
    val checked = targetResponse.sanityCheck(features, categoricalLabel = Option(true))
    val output = new OpWorkflow().setResultFeatures(checked).transform(textData)
    output.select(checked.name).count() shouldBe 12
    val meta = MinVarianceCheckerSummary.fromMetadata(checked.originStage.getMetadata().getSummaryMetadata())
    meta.dropped.size shouldBe 0
    meta.categoricalStats.size shouldBe 4
    meta.categoricalStats.foreach(_.contingencyMatrix("0").length shouldBe 2)
  }

  private def validateEstimatorOutput(outputColName: String, model: BinaryModel[RealNN, OPVector, OPVector],
    expectedFeaturesToDrop: Seq[String], label: String): Unit = {
    val metadata = model.getMetadata()
    val summary = MinVarianceCheckerSummary.fromMetadata(metadata.getSummaryMetadata())
    val toDropFeatureNames = summary.dropped
    val inFeatureNames = summary.names

    val featuresKept = OpVectorMetadata(outputColName, metadata).columns.length

    toDropFeatureNames should contain theSameElementsAs expectedFeaturesToDrop
    featuresKept equals inFeatureNames.toSet.diff(toDropFeatureNames.toSet).size
    summary.names.last shouldEqual label
    summary.featuresStatistics.max.last shouldBe 1 // this is max on label column
    summary.featuresStatistics.mean.last shouldBe 0.5 // this is mean on label column
  }

  private def validateTransformerOutput(
    outputColName: String,
    transformedData: DataFrame,
    expectedFeatNames: Seq[String],
    expectedCorrFeatNames: Seq[String],
    expectedFeaturesToDrop: Seq[String],
    expectedCorrFeatNamesIsNan: Seq[String]
  ): Unit = {
    transformedData.select(outputColName).collect().foreach { case Row(features: Vector) =>
      features.toArray.length equals
        (expectedCorrFeatNames.length + expectedCorrFeatNamesIsNan.length - expectedFeaturesToDrop.length)
    }

    val metadata: Metadata = getMetadata(outputColName, transformedData)
    val summary = MinVarianceCheckerSummary.fromMetadata(metadata.getSummaryMetadata())

    summary.names.slice(0, summary.names.size - 1) should
      contain theSameElementsAs expectedFeatNames
    summary.correlationsWLabel.nanCorrs should contain theSameElementsAs expectedCorrFeatNamesIsNan
    summary.correlationsWLabel.featuresIn should contain theSameElementsAs expectedCorrFeatNames
    summary.dropped should contain theSameElementsAs expectedFeaturesToDrop
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
