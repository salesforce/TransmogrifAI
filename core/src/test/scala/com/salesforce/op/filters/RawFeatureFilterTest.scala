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
import com.salesforce.op.features.{OPFeature, TransientFeature}
import com.salesforce.op.readers.DataFrameFieldNames
import com.salesforce.op.stages.impl.feature.HashAlgorithm
import com.salesforce.op.test.{Passenger, PassengerSparkFixtureTest}
import com.salesforce.op.utils.spark.RichDataset._
import com.twitter.algebird.Operators._
import org.apache.log4j.Level
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RawFeatureFilterTest extends FlatSpec with PassengerSparkFixtureTest with FiltersTestData {

  // loggingLevel(Level.INFO)
  conf.set("spark.kryo.registrationRequired", "true")
  conf.registerKryoClasses(Array(
    classOf[HistogramSummary],
    classOf[TextSummary],
    classOf[Array[scala.collection.immutable.Map[_, _]]],
    classOf[org.apache.spark.mllib.feature.HashingTF],
    Class.forName("com.salesforce.op.filters.HistogramSummary$$anon$7"),
    Class.forName("com.salesforce.op.filters.HistogramSummary$$anon$8"),
    Class.forName("com.salesforce.op.filters.HistogramSummary$$anon$9"),
    Class.forName("com.salesforce.op.filters.TextSummary$$anon$4"),
    Class.forName("com.salesforce.op.filters.TextSummary$$anon$5"),
    Class.forName("com.salesforce.op.filters.TextSummary$$anon$6"),
    Class.forName(
      "com.salesforce.op.filters.RawFeatureFilter$$anonfun$updateTextSummaries$1$1$$anonfun$41$$anonfun$apply$5"),
    Class.forName("com.salesforce.op.filters.RawFeatureFilter$$anonfun$updateTextSummaries$1$1$$anonfun$41"),
    Class.forName("com.salesforce.op.filters.RawFeatureFilter$$anonfun$updateTextSummaries$1$1")
  ))

  val trainAllDistributions = AllDistributions(
    responseDistributions = Map.empty,
    numericDistributions = trainSummaries,
    textDistributions = Map.empty)
  val scoreAllDistributions = Option(AllDistributions(
    responseDistributions = Map.empty,
    numericDistributions = scoreSummaries,
    textDistributions = Map.empty))

  Spec[RawFeatureFilter[_]] should "compute feature stats correctly" in {
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.8, Double.PositiveInfinity, 0.7, 1.0)
    val allFeatureInfo = filter.computeFeatureStats(passengersDataSet, features)
    val responseDistributions = allFeatureInfo.allDistributions.responseDistributions
    val predictorDistributions = allFeatureInfo.allDistributions.predictorDistributions

    // allFeatureInfo.responseSummaries.size shouldBe 1
    // allFeatureInfo.responseSummaries.headOption.map(_._2) shouldEqual Option(Summary(0, 1, 1, 2))
    responseDistributions.size shouldBe 1
    // allFeatureInfo.predictorSummaries.size shouldBe 12
    predictorDistributions.size shouldBe 12

    val survOpt = responseDistributions.get(survived.name -> None)
    val ageFOpt = predictorDistributions.get(age.name -> None)
    val strMapFOpt = predictorDistributions.get(stringMap.name -> Option("Female"))
    val strMapMOpt = predictorDistributions.get(stringMap.name -> Option("Male"))

    survOpt.nonEmpty shouldBe true
    ageFOpt.nonEmpty shouldBe true
    strMapFOpt.nonEmpty shouldBe true
    strMapMOpt.nonEmpty shouldBe true

    for {
      surv <- survOpt
      ageF <- ageFOpt
      strMapF <- strMapFOpt
      strMapM <- strMapMOpt
    } {
      withClue(s"Checking name for ${survived.name}") { surv.name shouldBe survived.name }
      withClue(s"Checking map key for ${survived.name}") { surv.key shouldBe None }
      withClue(s"Checking count for ${survived.name}") { surv.count shouldBe 6 }
      withClue(s"Checking nulls for ${survived.name}") { surv.nulls shouldBe 4 }
      withClue(s"Checking distribution sum for ${survived.name}") { surv.distribution.sum shouldBe 2 }

      withClue(s"Checking name for ${ageF.name}") { ageF.name shouldBe age.name }
      withClue(s"Checking map key for ${ageF.name}") { ageF.key shouldBe None }
      withClue(s"Checking count for ${ageF.name}") { ageF.count shouldBe 6 }
      withClue(s"Checking nulls for ${ageF.name}") { ageF.nulls shouldBe 2 }
      withClue(s"Checking distribution sum for ${ageF.name}") { ageF.distribution.sum shouldBe 4 }

      strMapF.name shouldBe stringMap.name
      if (strMapF.key.contains("Female")) strMapF.nulls shouldBe 3 else strMapF.nulls shouldBe 4

      strMapM.name shouldBe stringMap.name
      if (strMapM.key.contains("Male")) strMapM.nulls shouldBe 4 else strMapM.nulls shouldBe 3
    }
  }

  it should "correctly determine which features to exclude based on the stats of training fill rate" in {
    // only fill rate matters
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val (excludedTrainF, excludedTrainMK) =
      filter.getFeaturesToExclude(trainAllDistributions, scoreAllDistributions, Map.empty)

    withClue("Checking training fill rate dropped feature names") {
      excludedTrainF.toSet shouldEqual Set("B", "D")
    }
    withClue("Checking training fill rate dropped feature map keys") {
      excludedTrainMK.keySet shouldEqual Set("C")
    }
    withClue("Checking training fill rate dropped feature map key values") {
      excludedTrainMK.head._2 shouldEqual Set("2")
    }
  }

  it should "correctly determine which features to exclude based on the stats of training and scoring fill rate" in {
    // only fill rate matters

    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val (excludedBothF, excludedBothMK) =
      filter.getFeaturesToExclude(trainAllDistributions, scoreAllDistributions, Map.empty)

    withClue("Checking training and scoring fill rate dropped feature names") {
      excludedBothF.toSet shouldEqual Set("B", "D")
    }
    withClue("Checking training and scoring fill rate dropped feature map keys") {
      excludedBothMK.keySet shouldEqual Set("C")
    }
    withClue("Checking training and scoring fill rate dropped feature map key values") {
      excludedBothMK.head._2 shouldEqual Set("2")
    }
  }

  it should "correctly determine which features to exclude based on the stats of relative fill rate" in {
    // relative fill rate matters
    val filter2 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 0.5, Double.PositiveInfinity, 1.0, 1.0)
    val (excludedBothRelF, excludedBothRelMK) =
      filter2.getFeaturesToExclude(trainAllDistributions, scoreAllDistributions, Map.empty)

    withClue("Checking relative fill rate dropped feature names") {
      excludedBothRelF.toSet shouldEqual Set("A")
    }
    withClue("Checking relative fill rate dropped feature map keys") {
      excludedBothRelMK.isEmpty shouldBe true
    }
  }

  it should "correctly determine which features to exclude based on the stats of fill rate ratio" in {
    // relative fill ratio matters
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0, 2.0, 1.0, 1.0)
    val (excludedBothRelFR, excludedBothRelMKR) =
      filter4.getFeaturesToExclude(trainAllDistributions, scoreAllDistributions, Map.empty)

    withClue("Checking fill rate ratio dropped feature names") {
      excludedBothRelFR.toSet shouldEqual Set("D", "A", "B")
    }
    withClue("Checking fill rate ratio dropped feature map keys") {
      excludedBothRelMKR.isEmpty shouldBe true
    }

  }

  it should "correctly determine which features to exclude based on the stats of js distance" in {
    // js distance
    val filter3 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0, Double.PositiveInfinity, 0.5, 1.0)
    val (excludedBothDistF, excludedBothDistMK) =
      filter3.getFeaturesToExclude(trainAllDistributions, scoreAllDistributions, Map.empty)

    withClue("Checking JS divergence dropped feature names") {
      excludedBothDistF.toSet shouldEqual Set("D")
    }
    withClue("Checking JS divergence dropped feature map keys") {
      excludedBothDistMK.keySet shouldEqual Set("C")
    }
    withClue("Checking JS divergence dropped feature map key values") {
      excludedBothDistMK.head._2 shouldEqual Set("1")
    }
  }

  it should "correctly determine which features to exclude based on all the stats" in {
    // all
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.5, Double.PositiveInfinity, 0.5, 1.0)
    val (excludedBothAllF, excludedBothAllMK) =
      filter4.getFeaturesToExclude(trainAllDistributions, scoreAllDistributions, Map.empty)
    excludedBothAllF.toSet shouldEqual Set("A", "B", "C", "D")
    excludedBothAllMK.isEmpty shouldBe true
  }

  it should "correctly clean the dataframe returned and give the features to blacklist" in {
    val params = new OpParams()
    val survPred = survived.copy(isResponse = false)
    val features: Array[OPFeature] =
      Array(survPred, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.0, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.isEmpty shouldBe true
    filteredRawData.mapKeysToDrop.isEmpty shouldBe true
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields

    val filter1 = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData1 = filter1.generateFilteredRaw(features, params)
    filteredRawData1.featuresToDrop should contain theSameElementsAs Array(survPred)
    filteredRawData1.mapKeysToDrop should contain theSameElementsAs Map("numericMap" -> Set("Male"),
      "booleanMap" -> Set("Male"), "stringMap" -> Set("Male"))
    filteredRawData1.cleanedData.schema.fields.exists(_.name == survPred.name) shouldBe false
    filteredRawData1.cleanedData.collect(stringMap).
      foreach(m => if (m.nonEmpty) m.value.keySet shouldEqual Set("Female"))
  }

  it should "not drop response features" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.isEmpty shouldBe true
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields
    filteredRawData.cleanedData.collect(stringMap)
      .foreach(m => if (m.nonEmpty) m.value.keySet shouldEqual Set("Female"))
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

    val filter2 = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.1, 0.1, 2, 0.2, 0.9,
      protectedFeatures = Set(age.name, gender.name))
    val filteredRawData2 = filter2.generateFilteredRaw(features, params)
    filteredRawData2.featuresToDrop.toSet shouldEqual Set(height, weight, description, boarded)
    filteredRawData2.cleanedData.schema.fields.map(_.name) should contain theSameElementsAs
      Array(DataFrameFieldNames.KeyFieldName, survived.name, age.name, gender.name)
  }

  it should "not drop JS divergence-protected features based on JS divergence check" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, boardedTime, boardedTimeAsDateTime)
    val filter = new RawFeatureFilter(
      trainingReader = dataReader,
      scoreReader = Some(simpleReader),
      bins = 10,
      minFill = 0.0,
      maxFillDifference = 1.0,
      maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 0.0,
      maxCorrelation = 1.0,
      jsDivergenceProtectedFeatures = Set(boardedTime.name, boardedTimeAsDateTime.name))

    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.toSet shouldEqual Set(gender, height, weight, description, boarded)
    filteredRawData.cleanedData.schema.fields.map(_.name) should contain theSameElementsAs
      Seq(DataFrameFieldNames.KeyFieldName, survived.name, age.name, boardedTime.name, boardedTimeAsDateTime.name)
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

  private def nullLabelCorrelationTest(
    maxCorrelation: Double,
    expectedDropped: Seq[OPFeature],
    expectedMapKeys: Seq[String],
    expectedDroppedMapKeys: Map[String, Set[String]]
  ): Unit = {
    def getFilter(maxCorrelation: Double): RawFeatureFilter[Passenger] = new RawFeatureFilter(
      trainingReader = dataReader,
      scoreReader = Some(simpleReader),
      bins = 10,
      minFill = 0.0,
      maxFillDifference = 1.0,
      maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 1.0,
      maxCorrelation = maxCorrelation)

    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val FilteredRawData(df, dropped, droppedKeyValue, _) =
      getFilter(maxCorrelation).generateFilteredRaw(features, params)

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
