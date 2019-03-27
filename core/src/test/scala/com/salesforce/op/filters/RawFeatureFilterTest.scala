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

package com.salesforce.op.filters

import com.salesforce.op.OpParams
import com.salesforce.op.features.{FeatureDistributionType, OPFeature}
import com.salesforce.op.readers.DataFrameFieldNames
import com.salesforce.op.test.{Passenger, PassengerSparkFixtureTest}
import com.salesforce.op.utils.spark.RichDataset._
import com.twitter.algebird.Operators._
import org.junit.runner.RunWith
import org.scalatest.{Assertion, FlatSpec}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RawFeatureFilterTest extends FlatSpec with PassengerSparkFixtureTest with FiltersTestData {

  Spec[RawFeatureFilter[_]] should "correctly compute feature stats" in {
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.8,
      Double.PositiveInfinity, 0.7, 1.0)
    val allFeatureInfo = filter.computeFeatureStats(passengersDataSet, features, FeatureDistributionType.Training)

    allFeatureInfo.responseSummaries.size shouldBe 1
    allFeatureInfo.responseSummaries.headOption.map(_._2) shouldEqual Option(Summary(0, 1, 1, 2))
    allFeatureInfo.responseDistributions.size shouldBe 1
    allFeatureInfo.predictorSummaries.size shouldBe 12
    allFeatureInfo.predictorDistributions.size shouldBe 12

    val surv = allFeatureInfo.responseDistributions(0)
    surv.name shouldBe survived.name
    surv.key shouldBe None
    surv.count shouldBe 6
    surv.nulls shouldBe 4
    surv.distribution.sum shouldBe 2

    val ageF = allFeatureInfo.predictorDistributions.filter(_.name == age.name)(0)
    ageF.name shouldBe age.name
    ageF.key shouldBe None
    ageF.count shouldBe 6
    ageF.nulls shouldBe 2
    ageF.distribution.sum shouldBe 4

    val strMapF =
      allFeatureInfo.predictorDistributions.filter(d => d.name == stringMap.name && d.key == Option("Female"))(0)

    strMapF.name shouldBe stringMap.name
    if (strMapF.key.contains("Female")) strMapF.nulls shouldBe 3 else strMapF.nulls shouldBe 4

    val strMapM =
      allFeatureInfo.predictorDistributions.filter(d => d.name == stringMap.name && d.key == Option("Male"))(0)

    strMapM.name shouldBe stringMap.name
    if (strMapM.key.contains("Male")) strMapM.nulls shouldBe 4 else strMapM.nulls shouldBe 3
  }

  it should "correctly compute and store raw feature filter metrics when correlation is unavailable " +
    "and scoring distribution is unavailable" in {
    // empty correlation info
    // empty scoring distributions
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0,
      Double.PositiveInfinity, 1.0, 1.0)
    val (rawFeatureFilterMetrics, _, _, _) = filter.getFeaturesToExclude(trainSummaries, Seq.empty, Map.empty)
    rawFeatureFilterMetrics.map(_.trainingFillRate) shouldBe List(0.9, 0.0, 0.9, 0.05, 0.1, 0.05)
    rawFeatureFilterMetrics.map(_.trainingNullLabelAbsoluteCorr) shouldBe List.fill(6)(None)
    rawFeatureFilterMetrics.map(_.scoringFillRate) shouldBe List.fill(6)(None)
    rawFeatureFilterMetrics.map(_.jsDivergence) shouldBe List.fill(6)(None)
    rawFeatureFilterMetrics.map(_.fillRateDiff) shouldBe List.fill(6)(None)
    rawFeatureFilterMetrics.map(_.fillRatioDiff) shouldBe List.fill(6)(None)
  }

  it should "correctly compute and store raw feature filter metrics when correlation is unavailable and " +
    "scoring distribution is available" in {
    // empty correlation info
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0,
      Double.PositiveInfinity, 1.0, 1.0)
    val (rawFeatureFilterMetrics, _, _, _) = filter.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    rawFeatureFilterMetrics.map(_.trainingFillRate) shouldEqual List(0.9, 0.0, 0.9, 0.05, 0.1, 0.05)
    rawFeatureFilterMetrics.map(_.trainingNullLabelAbsoluteCorr) shouldEqual List.fill(6)(None)
    rawFeatureFilterMetrics.map(_.scoringFillRate) shouldEqual
      List(Some(0.2), Some(0.0), Some(0.9), Some(0.05), Some(0.0), Some(0.0))
    rawFeatureFilterMetrics.map(_.jsDivergence).dropRight(2) shouldEqual
      List(Some(0.0), Some(0.0), Some(1.0), Some(0.0))
    rawFeatureFilterMetrics.map(_.jsDivergence).drop(4).map { jsd =>
      jsd match {
        case Some(x) => x.isNaN
        case _ => false
      }
    } shouldEqual List(true, true)
    rawFeatureFilterMetrics.map(_.fillRateDiff) shouldEqual
      List(Some(0.7), Some(0.0), Some(0.0), Some(0.0), Some(0.1), Some(0.05))
    rawFeatureFilterMetrics.map(_.fillRatioDiff) shouldEqual
      List(Some(4.5), Some(Double.PositiveInfinity), Some(1.0), Some(1.0),
        Some(Double.PositiveInfinity), Some(Double.PositiveInfinity))
  }

  it should "correctly determine which features to exclude based on the stats of training fill rate" in {
    // only fill rate matters
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0,
      Double.PositiveInfinity, 1.0, 1.0)
    val (rawFeatureFilterMetrics, exclusionReasons, excludedTrainF, excludedTrainMK) =
      filter.getFeaturesToExclude(trainSummaries, Seq.empty, Map.empty)
    excludedTrainF.toSet shouldEqual Set("B", "D")
    excludedTrainMK.keySet shouldEqual Set("C")
    excludedTrainMK.head._2 shouldEqual Set("2")
    exclusionReasons.filter(_.trainingUnfilledState).map { _.name}.toSet shouldEqual Set("B", "C", "D")
  }

  it should "correctly determine which features to exclude based on the stats of training and scoring fill rate" in {
    // only fill rate matters

    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0,
      Double.PositiveInfinity, 1.0, 1.0)
    val (rawFeatureFilterMetrics, exclusionReasons, excludedBothF, excludedBothMK) =
      filter.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothF.toSet shouldEqual Set("B", "D")
    excludedBothMK.keySet shouldEqual Set("C")
    excludedBothMK.head._2 shouldEqual Set("2")
    exclusionReasons.filter(x => x.trainingUnfilledState || x.scoringUnfilledState).map { _.name}.toSet shouldEqual
      Set("B", "C", "D")
  }

  it should "correctly determine which features to exclude based on the stats of relative fill rate" in {
    // relative fill rate matters
    val filter2 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 0.5,
      Double.PositiveInfinity, 1.0, 1.0)
    val (rawFeatureFilterMetrics, exclusionReasons, excludedBothRelF, excludedBothRelMK) =
      filter2.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothRelF.toSet shouldEqual Set("A")
    excludedBothRelMK shouldBe empty
    exclusionReasons.filter(_.fillRateDiffMismatch).map { _.name}.toSet shouldEqual Set("A")
  }

  it should "correctly determine which features to exclude based on the stats of fill rate ratio" in {
    // relative fill ratio matters
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0, 2.0, 1.0, 1.0)
    val (rawFeatureFilterMetrics, exclusionReasons, excludedBothRelFR, excludedBothRelMKR) =
      filter4.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothRelFR.toSet shouldEqual Set("D", "A", "B")
    excludedBothRelMKR shouldBe empty
    exclusionReasons.filter(_.fillRatioDiffMismatch).map { _.name}.toSet shouldEqual Set("A", "B", "D")
  }

  it should "correctly determine which features to exclude based on the stats of js distance" in {
    // js distance
    val filter3 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0,
      Double.PositiveInfinity, 0.5, 1.0)
    val (rawFeatureFilterMetrics, exclusionReasons, excludedBothDistF, excludedBothDistMK) =
      filter3.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothDistF.isEmpty shouldEqual true
    excludedBothDistMK.keySet shouldEqual Set("C")
    excludedBothDistMK.head._2 shouldEqual Set("1")
    exclusionReasons.filter(_.excluded).map { _.name}.toSet shouldEqual Set("C")
  }

  it should "correctly determine which features to exclude based on all the stats" in {
    // all
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.5,
      Double.PositiveInfinity, 0.5, 1.0)
    val (rawFeatureMetrics, exclusionReasons, excludedBothAllF, excludedBothAllMK) =
      filter4.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothAllF.toSet shouldEqual Set("A", "B", "C", "D")
    excludedBothAllMK shouldBe empty
    exclusionReasons.filter(_.excluded).map { _.name }.toSet shouldEqual Set("A", "B", "C", "D")
  }

  it should "correctly clean the dataframe returned and give the features to blacklist" in {
    val params = new OpParams()
    val survPred = survived.copy(isResponse = false)
    val features: Array[OPFeature] =
      Array(survPred, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.0, 1.0,
      Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop shouldBe empty
    filteredRawData.mapKeysToDrop shouldBe empty
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields

    assertFeatureDistributions(filteredRawData, total = 26)

    val filter1 = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5,
      Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData1 = filter1.generateFilteredRaw(features, params)
    filteredRawData1.featuresToDrop should contain theSameElementsAs Array(survPred)
    filteredRawData1.mapKeysToDrop should contain theSameElementsAs Map(
      "numericMap" -> Set("Male"), "booleanMap" -> Set("Male"), "stringMap" -> Set("Male"))
    filteredRawData1.cleanedData.schema.fields.exists(_.name == survPred.name) shouldBe false
    filteredRawData1.cleanedData.collect(stringMap).foreach(m =>
      if (m.nonEmpty) m.value.keySet shouldEqual Set("Female"))
    assertFeatureDistributions(filteredRawData, total = 26)
  }

  it should "not drop response features" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5,
      Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop shouldBe empty
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields
    filteredRawData.cleanedData.collect(stringMap)
      .foreach(m => if (m.nonEmpty) m.value.keySet shouldEqual Set("Female"))
    assertFeatureDistributions(filteredRawData, total = 26)
  }

  it should "not drop protected features" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.1, 0.1, 2, 0.2, 0.9)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.toSet shouldEqual Set(age, gender, height, weight, description, boarded)
    filteredRawData.cleanedData.schema.fields.map(_.name) should contain theSameElementsAs
      Array(DataFrameFieldNames.KeyFieldName, survived.name)
    assertFeatureDistributions(filteredRawData, total = 14)

    val filter2 = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.1, 0.1, 2, 0.2, 0.9,
      protectedFeatures = Set(age.name, gender.name))
    val filteredRawData2 = filter2.generateFilteredRaw(features, params)
    filteredRawData2.featuresToDrop.toSet shouldEqual Set(height, weight, description, boarded)
    filteredRawData2.cleanedData.schema.fields.map(_.name) should contain theSameElementsAs
      Array(DataFrameFieldNames.KeyFieldName, survived.name, age.name, gender.name)
    assertFeatureDistributions(filteredRawData, total = 14)
  }

  it should "not drop JS divergence-protected features based on JS divergence check" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, boardedTime, boardedTimeAsDateTime)
    val filter = new RawFeatureFilter(
      trainingReader = dataReader,
      scoringReader = Some(simpleReader),
      bins = 10,
      minFill = 0.0,
      maxFillDifference = 1.0,
      maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 0.0,
      maxCorrelation = 1.0,
      jsDivergenceProtectedFeatures = Set(boardedTime.name, boardedTimeAsDateTime.name)
    )

    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.toSet shouldEqual Set(age, gender, height, weight, description, boarded)
    filteredRawData.cleanedData.schema.fields.map(_.name) should contain theSameElementsAs
      Seq(DataFrameFieldNames.KeyFieldName, survived.name, boardedTime.name, boardedTimeAsDateTime.name)
    assertFeatureDistributions(filteredRawData, total = 18)
  }

  it should "correctly drop features based on null-label leakage correlation greater than 0.9" in {
    val expectedDropped = Seq(boarded, weight, gender)
    val expectedMapKeys = Seq("Female", "Male")
    val expectedDroppedMapKeys = Map[String, Set[String]]()
    nullLabelCorrelationTest(0.9, expectedDropped, expectedMapKeys, expectedDroppedMapKeys)
  }

  it should "correctly drop features based on null-label leakage correlation greater than 0.6" in {
    val expectedDropped = Seq(boarded, weight, gender, age)
    val expectedMapKeys = Seq("Female", "Male")
    val expectedDroppedMapKeys = Map[String, Set[String]]()
    nullLabelCorrelationTest(0.6, expectedDropped, expectedMapKeys, expectedDroppedMapKeys)
  }

  it should "correctly drop features based on null-label leakage correlation greater than 0.4" in {
    val expectedDropped = Seq(boarded, weight, gender, age, description)
    val expectedMapKeys = Seq("Male")
    val expectedDroppedMapKeys = Map("booleanMap" -> Set("Female"), "stringMap" -> Set("Female"),
      "numericMap" -> Set("Female"))
    nullLabelCorrelationTest(0.4, expectedDropped, expectedMapKeys, expectedDroppedMapKeys)
  }

  it should "correctly drop features based on null-label leakage correlation greater than 0.3" in {
    val expectedDropped = Seq(boarded, weight, gender, age, description, booleanMap, numericMap, stringMap)
    // all the maps dropped
    val expectedDroppedMapKeys = Map[String, Set[String]]()
    nullLabelCorrelationTest(0.3, expectedDropped, Seq(), expectedDroppedMapKeys)
  }

  private def assertFeatureDistributions(fd: FilteredRawData, total: Int): Assertion = {
    fd.rawFeatureFilterResults.rawFeatureDistributions.length shouldBe total
    fd.trainingFeatureDistributions.foreach(_.`type` shouldBe FeatureDistributionType.Training)
    fd.trainingFeatureDistributions.length shouldBe total / 2
    fd.scoringFeatureDistributions.foreach(_.`type` shouldBe FeatureDistributionType.Scoring)
    fd.scoringFeatureDistributions.length shouldBe total / 2
    fd.trainingFeatureDistributions ++ fd.scoringFeatureDistributions shouldBe
      fd.rawFeatureFilterResults.rawFeatureDistributions


  }

  private def nullLabelCorrelationTest(
    maxCorrelation: Double,
    expectedDropped: Seq[OPFeature],
    expectedMapKeys: Seq[String],
    expectedDroppedMapKeys: Map[String, Set[String]]
  ): Unit = {
    def getFilter(maxCorrelation: Double): RawFeatureFilter[Passenger] = new RawFeatureFilter(
      trainingReader = dataReader,
      scoringReader = Some(simpleReader),
      bins = 10,
      minFill = 0.0,
      maxFillDifference = 1.0,
      maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 1.0,
      maxCorrelation = maxCorrelation)

    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filteredRawData@FilteredRawData(df, dropped, droppedKeyValue, _) =
      getFilter(maxCorrelation).generateFilteredRaw(features, params)

    assertFeatureDistributions(filteredRawData, total = 26)
    dropped should contain theSameElementsAs expectedDropped
    droppedKeyValue should contain theSameElementsAs expectedDroppedMapKeys

    df.schema.fields.map(_.name) should contain theSameElementsAs
      DataFrameFieldNames.KeyFieldName +: features.diff(dropped).map(_.name)
    if (expectedMapKeys.nonEmpty) {
      df.collect(booleanMap).map(_.value.keySet).reduce(_ + _) should contain theSameElementsAs expectedMapKeys
      df.collect(numericMap).map(_.value.keySet).reduce(_ + _) should contain theSameElementsAs expectedMapKeys
      df.collect(stringMap).map(_.value.keySet).reduce(_ + _) should contain theSameElementsAs expectedMapKeys
    } else {
      intercept[IllegalArgumentException] { df.collect(booleanMap) }
      intercept[IllegalArgumentException] { df.collect(numericMap) }
      intercept[IllegalArgumentException] { df.collect(stringMap) }
    }
  }
}
