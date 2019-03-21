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
import com.salesforce.op.features.{Feature, FeatureDistributionType, OPFeature}
import com.salesforce.op.features.types._
import com.salesforce.op.readers.{CustomReader, DataFrameFieldNames, ReaderKey}
import com.salesforce.op.test._
import com.salesforce.op.testkit._
import com.salesforce.op.testkit.RandomData
import com.salesforce.op.stages.impl.feature.OPMapVectorizerTestHelper.makeTernaryOPMapTransformer
import com.salesforce.op.utils.spark.RichDataset._
import com.twitter.algebird.Operators._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.junit.runner.RunWith
import org.scalatest.{Assertion, FlatSpec}
import org.scalatest.junit.JUnitRunner

import scala.reflect.runtime.universe.TypeTag

@RunWith(classOf[JUnitRunner])
class RawFeatureFilterTest extends FlatSpec with PassengerSparkFixtureTest with FiltersTestData {

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
  def generateRandomDfAndFeatures[
    F1 <: FeatureType : TypeTag,
    F2 <: FeatureType : TypeTag,
    F3 <: FeatureType : TypeTag,
    F4 <: FeatureType : TypeTag
  ](f1: RandomData[F1], f2: RandomData[F2], f3: RandomData[F3], f4: RandomData[F4], numRows: Int):
  (Dataset[Row], Feature[F1], Feature[F2], Feature[F3], Feature[F4])  = {

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

  it should "use a simple function to generate random data" in {
    val (testDf, f1, f2, f3, f4) = generateRandomDfAndFeatures[City, Country, PickList, Currency](
      RandomText.cities.withProbabilityOfEmpty(0.2),
      RandomText.countries.withProbabilityOfEmpty(0.2),
      RandomText.pickLists(domain = List("A", "B", "C", "D", "E", "F", "G", "H", "I")).withProbabilityOfEmpty(0.2),
      RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.2),
      1000
    )

    testDf.show(10)
  }

  it should "clean a dataframe filled with randomly generated data" in {
    // Define random generators that will be the same for training and scoring dataframes
    val cityGenerator = RandomText.cities.withProbabilityOfEmpty(0.2)
    val countryGenerator = RandomText.countries.withProbabilityOfEmpty(0.2)
    val pickListGenerator = RandomText.pickLists(domain = List("A", "B", "C", "D", "E", "F", "G", "H", "I"))
      .withProbabilityOfEmpty(0.2)
    val currencyGenerator = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, trainCity, trainCountry, trainPickList, trainCurrency) =
      generateRandomDfAndFeatures[City, Country, PickList, Currency](
        cityGenerator,
        countryGenerator,
        pickListGenerator,
        currencyGenerator.withProbabilityOfEmpty(0.2),
        1000
      )

    // Define the scoring dataframe (we can reuse the existing features so don't need to keep them)
    val (scoreDf, _, _, _, _) = generateRandomDfAndFeatures[City, Country, PickList, Currency](
        cityGenerator,
        countryGenerator,
        pickListGenerator,
        currencyGenerator.withProbabilityOfEmpty(1.0),
        1000
      )

    // Define the readers
    val trainReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(trainDf)
    }
    val scoreReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(scoreDf)
    }

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(trainCity, trainCountry, trainPickList, trainCurrency)
    val filter = new RawFeatureFilter(trainReader, Some(scoreReader), 10, 0.1, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)


    // Check that the only feature that was dropped was the currency feature
    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    filteredRawData.featuresToDrop.length shouldBe 1
    filteredRawData.featuresToDrop.head.name.startsWith("myF4") shouldBe true
  }

  it should "not remove any features when the training and scoring sets are identical" in {
    // Define random generators that will be the same for training and scoring dataframes
    val cityGenerator = RandomText.cities.withProbabilityOfEmpty(0.2)
    val countryGenerator = RandomText.countries.withProbabilityOfEmpty(0.2)
    val pickListGenerator = RandomText.pickLists(domain = List("A", "B", "C", "D", "E", "F", "G", "H", "I"))
      .withProbabilityOfEmpty(0.2)
    val currencyGenerator = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, trainCity, trainCountry, trainPickList, trainCurrency) =
    generateRandomDfAndFeatures[City, Country, PickList, Currency](
      cityGenerator,
      countryGenerator,
      pickListGenerator,
      currencyGenerator.withProbabilityOfEmpty(0.2),
      1000
    )

    // Define the readers
    val trainReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(trainDf)
    }
    val scoreReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(trainDf)
    }

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(trainCity, trainCountry, trainPickList, trainCurrency)
    val filter = new RawFeatureFilter(trainReader, Some(scoreReader), 10, 0.1, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)

    filteredRawData.featuresToDrop shouldBe empty
    filteredRawData.mapKeysToDrop shouldBe empty
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs
      trainReader.generateDataFrame(features).schema.fields

    assertFeatureDistributions(filteredRawData, total = features.length * 2)

    // Also check that the all the feature distributions are the same between the training and scoring sets
    filteredRawData.trainingFeatureDistributions.zip(filteredRawData.trainingFeatureDistributions).foreach{
      case (train, score) =>
        train.name shouldBe score.name
        train.key shouldBe score.key
        train.count shouldBe score.count
        train.nulls shouldBe score.nulls
        train.distribution shouldBe score.distribution
        train.summaryInfo shouldBe score.summaryInfo
    }
  }

  it should "correctly clean the dataframe due to min fill rates" in {
    // Define random generators that will be the same for training and scoring dataframes
    val stateGenerator = RandomText.states.withProbabilityOfEmpty(0.2)
    val pickListGenerator = RandomText.pickLists(domain = List("A", "B", "C", "D", "E", "F", "G", "H", "I"))
      .withProbabilityOfEmpty(0.2)
    val realMapGenerator = RandomMap.ofReals(RandomReal.logNormal[Real](mean = 10.0, sigma = 1.0), 0, 4)
      .withKeys ("customReal" +)
    val binaryMapGenerator = RandomMap.ofBinaries(probabilityOfSuccess = 0.25, 0, 4)
      .withKeys ("customBinary" +)

    val currencyGenerator = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0).withProbabilityOfEmpty(0.25)
    val c1 = currencyGenerator.limit(1000)
    val c2 = currencyGenerator.limit(1000)
    val c3 = currencyGenerator.limit(1000)
    val mapFeature = makeTernaryOPMapTransformer[Currency, CurrencyMap, Double](c1, c2, c3).asRaw(isResponse = false)

    // Define the training dataframe and the features (these should be the same between the training and scoring
    // dataframes since they point to columns with the same names)
    val (trainDf, trainState, trainPickList, trainRealMap, trainBinaryMap) =
    generateRandomDfAndFeatures[State, PickList, RealMap, BinaryMap](
      stateGenerator,
      pickListGenerator,
      realMapGenerator,
      binaryMapGenerator,
      1000
    )

    // Define the scoring dataframe (we can reuse the existing features so don't need to keep them)
    val (scoreDf, _, _, _, _) = generateRandomDfAndFeatures[City, Country, PickList, Currency](
      cityGenerator,
      countryGenerator,
      pickListGenerator,
      currencyGenerator.withProbabilityOfEmpty(1.0),
      1000
    )

    // Define the readers
    val trainReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(trainDf)
    }
    val scoreReader = new CustomReader[Row](ReaderKey.randomKey) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Row], Dataset[Row]] = Right(scoreDf)
    }

    val params = new OpParams()
    // We should be able to set the features to either be the train features or the score ones here
    val features: Array[OPFeature] = Array(trainCity, trainCountry, trainPickList, trainCurrency)
    val filter = new RawFeatureFilter(trainReader, Some(scoreReader), 10, 0.1, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)


    // Check that the only feature that was dropped was the currency feature
    // TODO: Add a check for the reason dropped once that information is passed on to the workflow
    filteredRawData.featuresToDrop.length shouldBe 1
    filteredRawData.featuresToDrop.head.name.startsWith("myF4") shouldBe true
  }

  it should "correctly clean the dataframe returned and give the features to blacklist" in {
    val params = new OpParams()
    val survPred = survived.copy(isResponse = false)
    val features: Array[OPFeature] =
      Array(survPred, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.0, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop shouldBe empty
    filteredRawData.mapKeysToDrop shouldBe empty
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields

    assertFeatureDistributions(filteredRawData, total = 26)

    val filter1 = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5, Double.PositiveInfinity, 1.0, 1.0)
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
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5, Double.PositiveInfinity, 1.0, 1.0)
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
    fd.featureDistributions.length shouldBe total
    fd.trainingFeatureDistributions.foreach(_.`type` shouldBe FeatureDistributionType.Training)
    fd.trainingFeatureDistributions.length shouldBe total / 2
    fd.scoringFeatureDistributions.foreach(_.`type` shouldBe FeatureDistributionType.Scoring)
    fd.scoringFeatureDistributions.length shouldBe total / 2
    fd.trainingFeatureDistributions ++ fd.scoringFeatureDistributions shouldBe fd.featureDistributions
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
