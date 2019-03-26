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

import com.salesforce.op.{OpParams, OpWorkflow}
import com.salesforce.op.features.{Feature, FeatureDistributionType, FeatureLike, OPFeature}
import com.salesforce.op.features.types._
import com.salesforce.op.readers.{CustomReader, DataFrameFieldNames, ReaderKey}
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.test._
import com.salesforce.op.testkit._
import com.salesforce.op.testkit.RandomData
import com.salesforce.op.stages.impl.feature.OPMapVectorizerTestHelper.makeTernaryOPMapTransformer
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.log4j.Level
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import com.twitter.algebird.Operators._
import org.junit.runner.RunWith
import org.scalatest.{Assertion, FlatSpec}
import org.scalatest.junit.JUnitRunner

import scala.reflect.runtime.universe.TypeTag

@RunWith(classOf[JUnitRunner])
class RawFeatureFilterTest extends FlatSpec with PassengerSparkFixtureTest with FiltersTestData {

  // loggingLevel(Level.INFO)

  // Our randomly generated data will generate feature names and corresponding map keys in this universe
  val featureUniverse = Set("myF1", "myF2", "myF3")
  val mapKeyUniverse = Set("f1", "f2", "f3")
  // Number of rows to use in randomly generated data sets
  val numRows = 1000

  Spec[RawFeatureFilter[_]] should "compute feature stats correctly" in {
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.8, Double.PositiveInfinity, 0.7, 1.0)
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

  it should "correctly determine which features to exclude based on the stats of training fill rate" in {
    // only fill rate matters
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val (excludedTrainF, excludedTrainMK) = filter.getFeaturesToExclude(trainSummaries, Seq.empty, Map.empty)
    excludedTrainF.toSet shouldEqual Set("B", "D")
    excludedTrainMK.keySet shouldEqual Set("C")
    excludedTrainMK.head._2 shouldEqual Set("2")
  }

  it should "correctly determine which features to exclude based on the stats of training and scoring fill rate" in {
    // only fill rate matters

    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val (excludedBothF, excludedBothMK) = filter.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothF.toSet shouldEqual Set("B", "D")
    excludedBothMK.keySet shouldEqual Set("C")
    excludedBothMK.head._2 shouldEqual Set("2")
  }

  it should "correctly determine which features to exclude based on the stats of relative fill rate" in {
    // relative fill rate matters
    val filter2 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 0.5, Double.PositiveInfinity, 1.0, 1.0)
    val (excludedBothRelF, excludedBothRelMK) = filter2.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothRelF.toSet shouldEqual Set("A")
    excludedBothRelMK shouldBe empty
  }

  it should "correctly determine which features to exclude based on the stats of fill rate ratio" in {
    // relative fill ratio matters
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0, 2.0, 1.0, 1.0)
    val (excludedBothRelFR, excludedBothRelMKR) =
      filter4.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothRelFR.toSet shouldEqual Set("D", "A", "B")
    excludedBothRelMKR shouldBe empty
  }

  it should "correctly determine which features to exclude based on the stats of js distance" in {
    // js distance
    val filter3 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0, Double.PositiveInfinity, 0.5, 1.0)
    val (excludedBothDistF, excludedBothDistMK) =
      filter3.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothDistF.isEmpty shouldEqual true
    excludedBothDistMK.keySet shouldEqual Set("C")
    excludedBothDistMK.head._2 shouldEqual Set("1")
  }

  it should "correctly determine which features to exclude based on all the stats" in {
    // all
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.5, Double.PositiveInfinity, 0.5, 1.0)
    val (excludedBothAllF, excludedBothAllMK) = filter4.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothAllF.toSet shouldEqual Set("A", "B", "C", "D")
    excludedBothAllMK shouldBe empty
  }

  it should "correctly clean the dataframe returned and give the features to blacklist" in {
    val params = new OpParams()
    val survPred = survived.copy(isResponse = false)
    val features: Array[OPFeature] =
      Array(survPred, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader),
      10, 0.0, 1.0, Double.PositiveInfinity, 1.0, 1.0, minScoringRows = 0)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop shouldBe empty
    filteredRawData.mapKeysToDrop shouldBe empty
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields

    assertFeatureDistributions(filteredRawData, total = 26)

    val filter1 = new RawFeatureFilter(dataReader, Some(simpleReader),
      10, 0.5, 0.5, Double.PositiveInfinity, 1.0, 1.0, minScoringRows = 0)
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
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader),
      10, 0.5, 0.5, Double.PositiveInfinity, 1.0, 1.0, minScoringRows = 0)
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
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader),
      10, 0.1, 0.1, 2, 0.2, 0.9, minScoringRows = 0)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.toSet shouldEqual Set(age, gender, height, weight, description, boarded)
    filteredRawData.cleanedData.schema.fields.map(_.name) should contain theSameElementsAs
      Array(DataFrameFieldNames.KeyFieldName, survived.name)
    assertFeatureDistributions(filteredRawData, total = 14)

    val filter2 = new RawFeatureFilter(dataReader, Some(simpleReader),
      10, 0.1, 0.1, 2, 0.2, 0.9, minScoringRows = 0,
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
      jsDivergenceProtectedFeatures = Set(boardedTime.name, boardedTimeAsDateTime.name),
      minScoringRows = 0
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

  /**
   * This test uses several data generators to generate data according to different distributions, makes a reader
   * corresponding to the generated dataframe, and then uses that as both the training and scoring reader in
   * RawFeatureFilter. Not only should no features be removed, but the training and scoring distributions should be
   * identical.
   */
  it should "not remove any features when the training and scoring sets are identical generated data" in {
    // Define random generators that will be the same for training and scoring dataframes
    val cityGenerator = RandomText.cities.withProbabilityOfEmpty(0.2)
    val countryGenerator = RandomText.countries.withProbabilityOfEmpty(0.2)
    val pickListGenerator = RandomText.pickLists(domain = List("A", "B", "C", "D", "E", "F", "G", "H", "I"))
      .withProbabilityOfEmpty(0.2)
    val currencyGenerator = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.2)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, trainCity, trainCountry, trainPickList, trainCurrency) =
    generateRandomDfAndFeatures[City, Country, PickList, Currency](
      cityGenerator, countryGenerator, pickListGenerator, currencyGenerator, numRows
    )

    // Define the readers
    val (trainReader, scoreReader) = makeReaders(trainDf, trainDf)

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(trainCity, trainCountry, trainPickList, trainCurrency)
    val filter = new RawFeatureFilter(trainReader, Some(scoreReader), 10, 0.4, 0.1, 1.0, 0.1, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)

    filteredRawData.featuresToDrop shouldBe empty
    filteredRawData.mapKeysToDrop shouldBe empty
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs
      trainReader.generateDataFrame(features).schema.fields

    assertFeatureDistributionEquality(filteredRawData, total = features.length * 2)
  }

  /**
   * This test generates three numeric generators with the same underlying distribution, but different fill rates.
   * Each generator corresponds to a different raw feature. Additionally, a single map feature is made from the three
   * raw features - each key contains the same data as the corresponding raw feature.
   *
   * Features c1, c2, and c3 are permuted between the training and scoring sets. In the training set, feature c2
   * has a 5% fill rate and should be removed. In the scoring set, map key f1 has a 5% fill rate so should be removed.
   * This test checks removal when only the training reader is used, and when both the training and scoring readers
   * are used.
   *
   */
  it should "correctly clean randomly generated map and non-map features due to min fill rate" in {
    // Define random generators that will be the same for training and scoring dataframes
    val currencyGenerator25 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.25)
    val currencyGenerator95 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.95)
    val currencyGenerator50 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.5)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, c1, c2, c3) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator25, currencyGenerator95, currencyGenerator50, numRows
    )
    val mapFeature = makeTernaryOPMapTransformer[Currency, CurrencyMap, Double](c1, c2, c3)
    // Need to make a raw version of this feature so that RawFeatureFilter will pick it up
    val mapFeatureRaw = mapFeature.asRaw(isResponse = false)
    val transformedTrainDf = new OpWorkflow().setResultFeatures(mapFeature).transform(trainDf)

    // Define the scoring dataframe (we can reuse the existing features so don't need to keep them)
    val (scoreDf, _, _, _) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator95, currencyGenerator50, currencyGenerator25, numRows
    )
    val transformedScoreDf = new OpWorkflow().setResultFeatures(mapFeature).transform(scoreDf)

    // Define the readers
    val (trainReader, scoreReader) = makeReaders(transformedTrainDf, transformedScoreDf)

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(c1, c2, c3, mapFeatureRaw)
    // Check that using the training reader only will result in the rarely filled features being removed
    val filter = new RawFeatureFilter(trainReader, None, 10, 0.1, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)

    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    checkDroppedFeatures(
      filteredRawData,
      mapFeatureRaw,
      featureUniverse = featureUniverse,
      expectedDroppedFeatures = Set(c2.name),
      mapKeyUniverse = mapKeyUniverse,
      expectedDroppedKeys = Set("f2")
    )

    // Check that using the scoring reader only will result in the rarely filled in both training and scoring sets
    // being removed
    val filterWithScoring = new RawFeatureFilter(trainReader, Some(scoreReader),
      10, 0.1, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawDataWithScoring = filterWithScoring.generateFilteredRaw(features, params)

    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    checkDroppedFeatures(
      filteredRawDataWithScoring,
      mapFeatureRaw,
      featureUniverse = featureUniverse,
      expectedDroppedFeatures = Set(c1.name, c2.name),
      mapKeyUniverse = mapKeyUniverse,
      expectedDroppedKeys = Set("f1", "f2")
    )

    // There should be 12 FeatureDistributions - training and scoring for 3 raw features, one map with three keys
    assertFeatureDistributions(filteredRawDataWithScoring, total = 12)
  }

  /**
   * This test generates three numeric generators with the same underlying distribution, but different fill rates.
   * Each generator corresponds to a different raw feature. Additionally, a single map feature is made from the three
   * raw features - each key contains the same data as the corresponding raw feature.
   *
   * Features c2 & c3 are switched between the training and scoring sets, so that they should have an absolute
   * fill rate difference of 0.6. The RawFeatureFilter is set up with a maximum absolute fill rate of 0.4 so both
   * c2 and c3 (as well as their corresponding map keys f2 & f3) should be removed.
   */
  it should "correctly clean the randomly generated map and non-map features due to max absolute fill rate " +
    "difference" in {
    // Define random generators that will be the same for training and scoring dataframes
    val currencyGenerator1 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.0)
    val currencyGenerator2 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.2)
    val currencyGenerator3 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.8)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, c1, c2, c3) =
    generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator1, currencyGenerator2, currencyGenerator3, numRows
    )
    val mapFeature = makeTernaryOPMapTransformer[Currency, CurrencyMap, Double](c1, c2, c3)
    // Need to make a raw version of this feature so that RawFeatureFilter will pick it up
    val mapFeatureRaw = mapFeature.asRaw(isResponse = false)
    val transformedTrainDf = new OpWorkflow().setResultFeatures(mapFeature).transform(trainDf)

    // Define the scoring dataframe (we can reuse the existing features so don't need to keep them)
    val (scoreDf, _, _, _) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator1, currencyGenerator3, currencyGenerator2, numRows
    )
    val transformedScoreDf = new OpWorkflow().setResultFeatures(mapFeature).transform(scoreDf)

    // Define the readers
    val (trainReader, scoreReader) = makeReaders(transformedTrainDf, transformedScoreDf)

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(c1, c2, c3, mapFeatureRaw)
    val filter = new RawFeatureFilter(trainReader, Some(scoreReader),
      10, 0.0, 0.4, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)

    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    checkDroppedFeatures(
      filteredRawData,
      mapFeatureRaw,
      featureUniverse = featureUniverse,
      expectedDroppedFeatures = Set(c2.name, c3.name),
      mapKeyUniverse = mapKeyUniverse,
      expectedDroppedKeys = Set("f2", "f3")
    )

    // There should be 12 FeatureDistributions - training and scoring for 3 raw features, one map with three keys
    assertFeatureDistributions(filteredRawData, total = 12)
  }

  /**
   * This test generates three numeric generators with the same underlying distribution, but different fill rates.
   * Each generator corresponds to a different raw feature. Additionally, a single map feature is made from the three
   * raw features - each key contains the same data as the corresponding raw feature.
   *
   * Features c2 & c3 are switched between the training and scoring sets, so that they should have an absolute
   * fill rate difference of 0.25, and a relative fill ratio difference of 6. The RawFeatureFilter is set up with a
   * maximum fill ratio difference of 4 so both c2 and c3 (as well as their corresponding map keys) should be removed.
   */
  it should "correctly clean the randomly generated map and non-map features due to max fill ratio " +
    "difference" in {
    // Define random generators that will be the same for training and scoring dataframes
    val currencyGenerator1 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.0)
    val currencyGenerator2 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.95)
    val currencyGenerator3 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.7)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, c1, c2, c3) =
    generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator1, currencyGenerator2, currencyGenerator3, numRows
    )
    val mapFeature = makeTernaryOPMapTransformer[Currency, CurrencyMap, Double](c1, c2, c3)
    // Need to make a raw version of this feature so that RawFeatureFilter will pick it up
    val mapFeatureRaw = mapFeature.asRaw(isResponse = false)
    val transformedTrainDf = new OpWorkflow().setResultFeatures(mapFeature).transform(trainDf)

    // Define the scoring dataframe (we can reuse the existing features so don't need to keep them)
    val (scoreDf, _, _, _) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator1, currencyGenerator3, currencyGenerator2, numRows
    )
    val transformedScoreDf = new OpWorkflow().setResultFeatures(mapFeature).transform(scoreDf)

    // Define the readers
    val (trainReader, scoreReader) = makeReaders(transformedTrainDf, transformedScoreDf)

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(c1, c2, c3, mapFeatureRaw)
    val filter = new RawFeatureFilter(trainReader, Some(scoreReader), 10, 0.0, 1.0, 4.0, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)

    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    checkDroppedFeatures(
      filteredRawData,
      mapFeatureRaw,
      featureUniverse = featureUniverse,
      expectedDroppedFeatures = Set(c2.name, c3.name),
      mapKeyUniverse = mapKeyUniverse,
      expectedDroppedKeys = Set("f2", "f3")
    )

    // There should be 12 FeatureDistributions - training and scoring for 3 raw features, one map with three keys
    assertFeatureDistributions(filteredRawData, total = 12)
  }

  /**
   * This test generates three numeric generators with the same underlying distribution, but different fill rates.
   * Each generator corresponds to a different raw feature. Additionally, a single map feature is made from the three
   * raw features - each key contains the same data as the corresponding raw feature.
   *
   * Features c1 & c3 are switched between the training and scoring sets, so they should have a very large JS
   * divergence (practically, 1.0). The RawFeatureFilter is set up with a maximum JS divergence of 0.8, so both
   * f1 and f3 (as well as their corresponding map keys) should be removed.
   */
  it should "correctly clean the randomly generated map and non-map features due to JS divergence" in {
    // Define random generators that will be the same for training and scoring dataframes
    val currencyGenerator1 = RandomReal.logNormal[Currency](mean = 1.0, sigma = 5.0).withProbabilityOfEmpty(0.1)
    val currencyGenerator2 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 5.0)
    val currencyGenerator3 = RandomReal.logNormal[Currency](mean = 1000.0, sigma = 5.0).withProbabilityOfEmpty(0.1)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, c1, c2, c3) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator1, currencyGenerator2, currencyGenerator3, numRows
    )
    val mapFeature = makeTernaryOPMapTransformer[Currency, CurrencyMap, Double](c1, c2, c3)
    // Need to make a raw version of this feature so that RawFeatureFilter will pick it up
    val mapFeatureRaw = mapFeature.asRaw(isResponse = false)
    val transformedTrainDf = new OpWorkflow().setResultFeatures(mapFeature).transform(trainDf)

    // Define the scoring dataframe (we can reuse the existing features so don't need to keep them)
    val (scoreDf, _, _, _) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator3, currencyGenerator2, currencyGenerator1, numRows
    )
    val transformedScoreDf = new OpWorkflow().setResultFeatures(mapFeature).transform(scoreDf)

    // Define the readers
    val (trainReader, scoreReader) = makeReaders(transformedTrainDf, transformedScoreDf)

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(c1, c2, c3, mapFeatureRaw)
    val filter = new RawFeatureFilter(trainReader, Some(scoreReader), 10, 0.1, 1.0, Double.PositiveInfinity, 0.8, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)

    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    checkDroppedFeatures(
      filteredRawData,
      mapFeatureRaw,
      featureUniverse = featureUniverse,
      expectedDroppedFeatures = Set(c1.name, c3.name),
      mapKeyUniverse = mapKeyUniverse,
      expectedDroppedKeys = Set("f1", "f3")
    )

    // There should be 12 FeatureDistributions - training and scoring for 3 raw features, one map with three keys
    assertFeatureDistributions(filteredRawData, total = 12)
  }

  it should "not drop protected raw features or response features from generated data" in {
    // Define random generators that will be the same for training and scoring dataframes
    val currencyGenerator25 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.25)
    val currencyGenerator95 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.95)
    val currencyGenerator50 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.5)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, c1, c2, c3) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator25, currencyGenerator95, currencyGenerator50, numRows
    )
    val mapFeature = makeTernaryOPMapTransformer[Currency, CurrencyMap, Double](c1, c2, c3)
    // Need to make a raw version of this feature so that RawFeatureFilter will pick it up
    val mapFeatureRaw = mapFeature.asRaw(isResponse = false)
    val transformedTrainDf = new OpWorkflow().setResultFeatures(mapFeature).transform(trainDf)

    // Define the scoring dataframe (we can reuse the existing features so don't need to keep them)
    val (scoreDf, _, _, _) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator95, currencyGenerator50, currencyGenerator25, numRows
    )
    val transformedScoreDf = new OpWorkflow().setResultFeatures(mapFeature).transform(scoreDf)

    // Define the readers
    val (trainReader, scoreReader) = makeReaders(transformedTrainDf, transformedScoreDf)

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(c1, c2, c3, mapFeatureRaw)
    // Check that using the scoring reader only will result in the rarely filled in both training and scoring sets
    // being removed, except for the protected feature that would normally be removed
    val filterWithProtected = new RawFeatureFilter(trainReader, Some(scoreReader),
      bins = 10,
      minFill = 0.1,
      maxFillDifference = 1.0,
      maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 1.0,
      maxCorrelation = 1.0,
      protectedFeatures = Set(c1.name)
    )
    val filteredRawData = filterWithProtected.generateFilteredRaw(features, params)

    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    checkDroppedFeatures(
      filteredRawData,
      mapFeatureRaw,
      featureUniverse = featureUniverse,
      expectedDroppedFeatures = Set(c2.name),
      mapKeyUniverse = mapKeyUniverse,
      expectedDroppedKeys = Set("f1", "f2")
    )

    // There should be 12 FeatureDistributions - training and scoring for 3 raw features, one map with three keys
    assertFeatureDistributions(filteredRawData, total = 12)

    val filter = new RawFeatureFilter(trainReader, Some(scoreReader),
      bins = 10,
      minFill = 0.1,
      maxFillDifference = 1.0,
      maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 1.0,
      maxCorrelation = 1.0
    )
    val featuresWithResponse: Array[OPFeature] = Array(c1.copy(isResponse = true), c2, c3, mapFeatureRaw)
    val filteredRawDataWithResponse = filter.generateFilteredRaw(featuresWithResponse, params)

    checkDroppedFeatures(
      filteredRawDataWithResponse,
      mapFeatureRaw,
      featureUniverse = featureUniverse,
      expectedDroppedFeatures = Set(c2.name),
      mapKeyUniverse = mapKeyUniverse,
      expectedDroppedKeys = Set("f1", "f2")
    )

    // There should be 12 FeatureDistributions - training and scoring for 3 raw features, one map with three keys
    assertFeatureDistributions(filteredRawDataWithResponse, total = 12)
  }

  // TODO: Add a way to protect map keys from removal?
  /**
   * This test generates three numeric generators with very different underlying distributions. Each generator
   * corresponds to a different raw feature. Additionally, a single map feature is made from the three raw features -
   * each key contains the same data as the corresponding raw feature.
   *
   * Features c1 & c3 are switched between the training and scoring sets, so they should have a very large JS
   * divergence (practically, 1.0). The RawFeatureFilter is set up with a maximum JS divergence of 0.8, so both
   * c1 and c3 (as well as their corresponding map keys) should be removed, but c3 is added to a list of features
   * protected from JS divergence removal.
   */
  it should "not drop JS divergence-protected features based on JS divergence check with generated data" in {
    // Define random generators that will be the same for training and scoring dataframes
    val currencyGenerator1 = RandomReal.logNormal[Currency](mean = 1.0, sigma = 5.0).withProbabilityOfEmpty(0.1)
    val currencyGenerator2 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 5.0).withProbabilityOfEmpty(0.1)
    val currencyGenerator3 = RandomReal.logNormal[Currency](mean = 1000.0, sigma = 5.0).withProbabilityOfEmpty(0.1)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, c1, c2, c3) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator1, currencyGenerator2, currencyGenerator3, numRows
    )
    val mapFeature = makeTernaryOPMapTransformer[Currency, CurrencyMap, Double](c1, c2, c3)
    // Need to make a raw version of this feature so that RawFeatureFilter will pick it up
    val mapFeatureRaw = mapFeature.asRaw(isResponse = false)
    val transformedTrainDf = new OpWorkflow().setResultFeatures(mapFeature).transform(trainDf)

    // Define the scoring dataframe (we can reuse the existing features so don't need to keep them)
    val (scoreDf, _, _, _) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator3, currencyGenerator2, currencyGenerator1, numRows
    )
    val transformedScoreDf = new OpWorkflow().setResultFeatures(mapFeature).transform(scoreDf)

    // Define the readers
    val (trainReader, scoreReader) = makeReaders(transformedTrainDf, transformedScoreDf)

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(c1, c2, c3, mapFeatureRaw)
    val filter = new RawFeatureFilter(trainReader, Some(scoreReader),
      bins = 10,
      minFill = 0.1,
      maxFillDifference = 1.0,
      maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 0.8,
      maxCorrelation = 1.0,
      jsDivergenceProtectedFeatures = Set(c3.name)
    )
    val filteredRawData = filter.generateFilteredRaw(features, params)

    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    checkDroppedFeatures(
      filteredRawData,
      mapFeatureRaw,
      featureUniverse = featureUniverse,
      expectedDroppedFeatures = Set(c1.name),
      mapKeyUniverse = mapKeyUniverse,
      expectedDroppedKeys = Set("f1", "f3")
    )

    // There should be 12 FeatureDistributions - training and scoring for 3 raw features, one map with three keys
    assertFeatureDistributions(filteredRawData, total = 12)
  }

  /**
   * This test generates three numeric generators where ach generator corresponds to a different raw feature.
   * Additionally, a single map feature is made from the three raw features - each key contains the same data
   * as the corresponding raw feature.
   *
   * A binary label is generated with a perfect relationship to feature c2 - if it is empty then the label is 0,
   * otherwise it is 1. Therefore feature c2 (and its corresponding map key) should be removed by the correlation
   * check between a raw feature's null indicator and the label.
   */
  it should "correctly drop features based on null-label correlations with generated data" in {
    // Define random generators that will be the same for training and scoring dataframes
    val currencyGenerator1 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.3)
    val currencyGenerator2 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.3)
    val currencyGenerator3 = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.3)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, c1, c2, c3) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator1, currencyGenerator2, currencyGenerator3, numRows
    )
    val mapFeature = makeTernaryOPMapTransformer[Currency, CurrencyMap, Double](c1, c2, c3)
    // Need to make a raw version of this feature so that RawFeatureFilter will pick it up
    val mapFeatureRaw = mapFeature.asRaw(isResponse = false)

    // Construct a label that we know is directly correlated to the currency data
    val labelTransformer = new UnaryLambdaTransformer[Currency, RealNN](operationName = "labelFunc",
      transformFn = r => r.value match {
        case Some(v) => RealNN(1.0)
        case _ => RealNN(0.0)
      }
    )
    val labelData = labelTransformer.setInput(c2).getOutput().asInstanceOf[Feature[RealNN]].copy(isResponse = true)
    val labelDataRaw = labelData.asRaw(isResponse = true)
    val transformedTrainDf = new OpWorkflow().setResultFeatures(mapFeature, labelData).transform(trainDf)

    // Define the scoring dataframe (we can reuse the existing features so don't need to keep them)
    val (scoreDf, _, _, _) = generateRandomDfAndFeatures[Currency, Currency, Currency](
      currencyGenerator1, currencyGenerator2, currencyGenerator3, numRows
    )
    val transformedScoreDf = new OpWorkflow().setResultFeatures(mapFeature, labelData).transform(scoreDf)

    // Define the readers
    val (trainReader, scoreReader) = makeReaders(transformedTrainDf, transformedScoreDf)

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(c1, c2, c3, mapFeatureRaw, labelDataRaw)
    val filter = new RawFeatureFilter(trainReader, Some(scoreReader), 10, 0.1, 1.0, Double.PositiveInfinity, 1.0, 0.8)
    val filteredRawData = filter.generateFilteredRaw(features, params)

    // TODO: check that filter.getFeaturesToExclude contains the correlation exclusions too
    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    checkDroppedFeatures(
      filteredRawData,
      mapFeatureRaw,
      featureUniverse = featureUniverse ++ Set(labelData.name),
      expectedDroppedFeatures = Set(c2.name),
      mapKeyUniverse = mapKeyUniverse,
      expectedDroppedKeys = Set("f2")
    )

    // There should be 14 FeatureDistributions - training and scoring for 4 raw features, one map with three keys
    assertFeatureDistributions(filteredRawData, total = 14)
  }

  /**
   * Generates a random dataframe and OPFeatures from supplied data generators and their types. The names of the
   * columns of the dataframe are fixed to be myF1, myF2, myF3, and myF4 so that the same OPFeatures can be used to
   * refer to columns in either dataframe.
   *
   * @param f1        Random data generator for feature 1 (type F1)
   * @param f2        Random data generator for feature 2 (type F2)
   * @param f3        Random data generator for feature 3 (type F3)
   * @param numRows   Number of rows to generate
   * @tparam F1       Type of feature 1
   * @tparam F2       Type of feature 2
   * @tparam F3       Type of feature 3
   * @return          Tuple containing the generated dataframe and each individual OPFeature
   */
  def generateRandomDfAndFeatures[F1 <: FeatureType : TypeTag,
    F2 <: FeatureType : TypeTag,
    F3 <: FeatureType : TypeTag
  ](f1: RandomData[F1], f2: RandomData[F2], f3: RandomData[F3], numRows: Int):
  (Dataset[Row], Feature[F1], Feature[F2], Feature[F3]) = {

    val f1Data = f1.limit(numRows)
    val f2Data = f2.limit(numRows)
    val f3Data = f3.limit(numRows)

    // Combine the data into a single tuple for each row
    val generatedTrainData: Seq[(F1, F2, F3)] = f1Data.zip(f2Data).zip(f3Data).map {
      case ((a, b), c) => (a, b, c)
    }

    TestFeatureBuilder[F1, F2, F3]("myF1", "myF2", "myF3", generatedTrainData)
  }

  /**
   * Generates a random dataframe and OPFeatures from supplied data generators and their types. The names of the
   * columns of the dataframe are fixed to be myF1, myF2, myF3, and myF4 so that the same OPFeatures can be used to
   * refer to columns in either dataframe.
   *
   * @param f1        Random data generator for feature 1 (type F1)
   * @param f2        Random data generator for feature 2 (type F2)
   * @param f3        Random data generator for feature 3 (type F3)
   * @param f4        Random data generator for feature 4 (type F4)
   * @param numRows   Number of rows to generate
   * @tparam F1       Type of feature 1
   * @tparam F2       Type of feature 2
   * @tparam F3       Type of feature 3
   * @tparam F4       Type of feature 4
   * @return          Tuple containing the generated dataframe and each individual OPFeature
   */
  def generateRandomDfAndFeatures[F1 <: FeatureType : TypeTag,
    F2 <: FeatureType : TypeTag,
    F3 <: FeatureType : TypeTag,
    F4 <: FeatureType : TypeTag
  ](f1: RandomData[F1], f2: RandomData[F2], f3: RandomData[F3], f4: RandomData[F4], numRows: Int):
  (Dataset[Row], Feature[F1], Feature[F2], Feature[F3], Feature[F4]) = {

    val f1Data = f1.limit(numRows)
    val f2Data = f2.limit(numRows)
    val f3Data = f3.limit(numRows)
    val f4Data = f4.limit(numRows)

    // Combine the data into a single tuple for each row
    val generatedTrainData: Seq[(F1, F2, F3, F4)] = f1Data.zip(f2Data).zip(f3Data).zip(f4Data).map {
      case (((a, b), c), d) => (a, b, c, d)
    }

    TestFeatureBuilder[F1, F2, F3, F4]("myF1", "myF2", "myF3", "myF4", generatedTrainData)
  }

  private def assertFeatureDistributions(fd: FilteredRawData, total: Int): Assertion = {
    fd.featureDistributions.length shouldBe total
    fd.trainingFeatureDistributions.foreach(_.`type` shouldBe FeatureDistributionType.Training)
    fd.trainingFeatureDistributions.length shouldBe total / 2
    fd.scoringFeatureDistributions.foreach(_.`type` shouldBe FeatureDistributionType.Scoring)
    fd.scoringFeatureDistributions.length shouldBe total / 2
    fd.trainingFeatureDistributions ++ fd.scoringFeatureDistributions shouldBe fd.featureDistributions
  }

  private def assertFeatureDistributionEquality(fd: FilteredRawData, total: Int): Unit = {
    fd.featureDistributions.length shouldBe total
    fd.trainingFeatureDistributions.zip(fd.scoringFeatureDistributions).foreach {
      case (train, score) =>
        train.name shouldBe score.name
        train.key shouldBe score.key
        train.count shouldBe score.count
        train.nulls shouldBe score.nulls
        train.distribution shouldBe score.distribution
        train.summaryInfo shouldBe score.summaryInfo
    }
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
      maxCorrelation = maxCorrelation,
      minScoringRows = 0)

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

  /**
   * Defines readers in terms of datasets (in these tests, already created by feature generators)
   *
   * @param trainDf   Training dataframe
   * @param scoreDf   Scoring dataframe
   * @return          Tuple of (trainingReader, scoringReader)
   */
  private def makeReaders(trainDf: Dataset[Row], scoreDf: Dataset[Row]): (CustomReader[Row], CustomReader[Row]) = {
    val trainReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(trainDf)
    }
    val scoreReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(scoreDf)
    }

    (trainReader, scoreReader)
  }

  // TODO: Expand scope to take multiple map types, and/or type parameters for maps of different types
  /**
   * Automates various checks on whether features are removed from the cleaned dataframe produced by RawFeatureFilter.
   * Right now, it is specialized to accept just one map type feature, which is hardcoded to be a CurrencyMap based
   * on current tests.
   *
   * @param filteredRawData         FilteredRawData object prdduced by RawFeatureFilter
   * @param mapFeatureRaw           Name of raw map feature to check keys on
   * @param featureUniverse         Set of raw feature names you start with
   * @param expectedDroppedFeatures Expected set of raw feature names to be dropped
   * @param mapKeyUniverse          Set of map keys in mapFeatureRaw you start with
   * @param expectedDroppedKeys     Expected set of map keys to be dropped
   */
  private def checkDroppedFeatures(
    filteredRawData: FilteredRawData,
    mapFeatureRaw: FeatureLike[CurrencyMap],
    featureUniverse: Set[String],
    expectedDroppedFeatures: Set[String],
    mapKeyUniverse: Set[String],
    expectedDroppedKeys: Set[String]
  ): Unit = {
    // Check that we drop one feature, as well as its corresponding map key
    filteredRawData.featuresToDrop.length shouldBe expectedDroppedFeatures.size
    filteredRawData.mapKeysToDrop.keySet.size shouldBe 1
    filteredRawData.mapKeysToDrop.values.head.size shouldBe expectedDroppedKeys.size

    filteredRawData.featuresToDrop.map(_.name) should contain theSameElementsAs expectedDroppedFeatures
    filteredRawData.mapKeysToDrop.head._2 should contain theSameElementsAs expectedDroppedKeys

    // Check the actual filtered dataframe schemas
    featureUniverse.foreach(f => {
      filteredRawData.cleanedData.schema.fields.exists(_.name == f) shouldBe !expectedDroppedFeatures.contains(f)
    })
    filteredRawData.cleanedData.collect(mapFeatureRaw)
      .foreach(_.value.keySet.intersect(expectedDroppedKeys) shouldBe Set.empty)
  }
}
