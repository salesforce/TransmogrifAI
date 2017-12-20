/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.MetadataParam
import com.salesforce.op.stages.base.binary.BinaryModel
import com.salesforce.op.stages.impl.feature.RealNNVectorizer
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.{DataFrame, Row}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FlatSpec

case class SanityCheckDataTest
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

@RunWith(classOf[JUnitRunner])
class SanityCheckerTest extends FlatSpec with TestSparkContext {

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
        indicatorGroup = None,
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


  Spec[SanityChecker] should "remove trouble features" in {
    val checked = targetLabel.sanityCheck(featureVector,
      maxCorrelation = 0.99, minVariance = 0.0, checkSample = 1.0, removeBadFeatures = true)
    val outputColName = checked.name

    checked.originStage shouldBe a[SanityChecker]

    val model = checked.originStage.asInstanceOf[SanityChecker].fit(testData)
    val featuresToDrop = Seq(featureNames(0), featureNames(3), featureNames(4))
    validateEstimatorOutput(outputColName, model, featuresToDrop, targetLabel.name)

    val transformedData = model.transform(testData)
    validateTransformerOutput(outputColName, transformedData, expectedCorrFeatNames,
      featuresToDrop, expectedCorrFeatNamesIsNan)
  }

  it should "not allow setting invalid param values" in {
    val checker = new SanityChecker()
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
    val checker = new SanityChecker()
    checker.getOrDefault(checker.sampleLowerLimit) shouldBe SanityChecker.SampleLowerLimit
    checker.getOrDefault(checker.sampleUpperLimit) shouldBe SanityChecker.SampleUpperLimit
    checker.getOrDefault(checker.checkSample) shouldBe SanityChecker.CheckSample
    checker.getOrDefault(checker.maxCorrelation) shouldBe SanityChecker.MaxCorrelation
    checker.getOrDefault(checker.minVariance) shouldBe SanityChecker.MinVariance
    checker.getOrDefault(checker.minCorrelation) shouldBe SanityChecker.MinCorrelation
    checker.getOrDefault(checker.correlationType) shouldBe CorrelationType.Pearson
    checker.getOrDefault(checker.removeBadFeatures) shouldBe SanityChecker.RemoveBadFeatures
  }

  it should "consume the output from an OP vectorizer and remove trouble features accordingly" in {
    val sanityChecker = new SanityChecker()

    val outputColName = sanityChecker
      .setMaxCorrelation(.99)
      .setMinVariance(0.0)
      .setCheckSample(1.0)
      .setRemoveBadFeatures(true)
      .setInput(targetLabel, featureVector).getOutput().name

    val model = sanityChecker.fit(testData)
    validateEstimatorOutput(outputColName, model, featuresToDrop, targetLabel.name)

    val transformedData = model.transform(testData)
    val metadata: Metadata = getMetadata(outputColName, transformedData)
    val summary = SanityCheckerSummary.fromMetadata(metadata.getSummaryMetadata())
    val outputColumns = OpVectorMetadata(transformedData.schema(outputColName))
    targetLabel.name +: testMetadata.columns.map(_.makeColName()) should
      contain theSameElementsAs summary.names
    outputColumns.columns.length + summary.dropped.length shouldEqual testMetadata.columns.length

    validateTransformerOutput(outputColName, transformedData, expectedCorrFeatNames,
      featuresToDrop, expectedCorrFeatNamesIsNan)
  }


  it should "not remove trouble features" in {

    val sanityChecker = new SanityChecker()

    val outputColName = sanityChecker
      .setMaxCorrelation(.99)
      .setMinVariance(0.0)
      .setCheckSample(1.0)
      .setInput(targetLabel, featureVector).getOutput().name

    val model = sanityChecker.fit(testData)
    validateEstimatorOutput(outputColName, model, Seq(), targetLabel.name)

    val transformedData = model.transform(testData)

    validateTransformerOutput(outputColName, transformedData, expectedCorrFeatNames,
      Seq(), expectedCorrFeatNamesIsNan)
  }

  it should "fail when not enough data to run" in {
    val sanityChecker = new SanityChecker()

    val outputColName = sanityChecker
      .setMaxCorrelation(.99)
      .setMinVariance(0.0)
      .setCheckSample(0.000001)
      .setRemoveBadFeatures(true)
      .setInput(targetLabel, featureVector).getOutput().name

    val result = sanityChecker.fit(testData)
    succeed
  }

  it should "fail when features are defined incorrectly" in {
    the[RuntimeException] thrownBy {
      new SanityChecker().setInput(targetLabel.copy(isResponse = true), featureVector.copy(isResponse = true))
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
    val sanityChecker = new SanityChecker()
      .setInput(f1r, f2)
      .setRemoveBadFeatures(true)
      .setCheckSample(1.0)
    val output = sanityChecker.getOutput()
    assertThrows[IllegalArgumentException] {
      sanityChecker.fit(data).transform(data)
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
    val sanityChecker = new SanityChecker()
      .setInput(f1r, f2)
      .setRemoveBadFeatures(true)
      .setCheckSample(0.99999)
      .setSampleLowerLimit(0)
      .setSampleUpperLimit(0)

    the[IllegalArgumentException] thrownBy {
      sanityChecker.fit(data)
    } should have message "requirement failed: Nothing has been added to this summarizer."
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

    def getFeatureCorrelationFor(sanityChecker: SanityChecker): Double = {
      val model = sanityChecker.fit(dataVectorized)

      val summaryMeta = model.getMetadata().getSummaryMetadata()
      val correlations = summaryMeta
        .getMetadata(SanityCheckerNames.CorrelationsWLabel)
        .getDoubleArray(SanityCheckerNames.Values)
      correlations(0)
    }

    val sanityCheckerSpearman = new SanityChecker()
      .setInput(label, vectorized)
      .setCorrelationType(CorrelationType.Spearman)
      .setCheckSample(0.99999)

    val sanityCheckerPearson = new SanityChecker()
      .setInput(label, vectorized)
      .setCorrelationType(CorrelationType.Pearson)
      .setCheckSample(0.99999)

    assert(
      getFeatureCorrelationFor(sanityCheckerSpearman) > getFeatureCorrelationFor(sanityCheckerPearson)
    )
  }

  it should "throw an error if all of the features are removed" in {
    val sanityChecker = new SanityChecker()

    val outputColName = sanityChecker
      .setMaxCorrelation(.000001)
      .setMinVariance(1000)
      .setCheckSample(0.999999)
      .setRemoveBadFeatures(true)
      .setInput(targetLabel, featureVector).getOutput().name

    the[RuntimeException] thrownBy {
      sanityChecker.fit(testData)
    } should have message
      "requirement failed: The sanity checker has dropped all of your features, check your input data quality"
  }

  private def validateEstimatorOutput(outputColName: String, model: BinaryModel[RealNN, OPVector, OPVector],
    expectedFeaturesToDrop: Seq[String], label: String): Unit = {
    val metadata = model.getMetadata()
    val summary = SanityCheckerSummary.fromMetadata(metadata.getSummaryMetadata())
    val toDropFeatureNames = summary.dropped
    val inFeatureNames = summary.names

    val featuresKept = OpVectorMetadata(outputColName, metadata).columns.length

    toDropFeatureNames should contain theSameElementsAs expectedFeaturesToDrop
    featuresKept equals inFeatureNames.toSet.diff(toDropFeatureNames.toSet).size
    summary.names.last shouldEqual label
    summary.featuresStatistics.max.last shouldBe 1 // this is max on label column
    summary.featuresStatistics.mean.last shouldBe 0.5 // this is mean on label column
  }

  private def validateTransformerOutput(outputColName: String, transformedData: DataFrame,
    expectedCorrFeatNames: Seq[String], expectedFeaturesToDrop: Seq[String],
    expectedCorrFeatNamesIsNan: Seq[String]): Unit = {
    transformedData.select(outputColName).collect().foreach { case Row(features: Vector) =>
      features.toArray.length equals
        (expectedCorrFeatNames.length + expectedCorrFeatNamesIsNan.length - expectedFeaturesToDrop.length)
    }

    val metadata: Metadata = getMetadata(outputColName, transformedData)
    val summary = SanityCheckerSummary.fromMetadata(metadata.getSummaryMetadata())

    summary.names.slice(0, summary.names.size - 1) should
      contain theSameElementsAs expectedCorrFeatNames ++ expectedCorrFeatNamesIsNan
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
