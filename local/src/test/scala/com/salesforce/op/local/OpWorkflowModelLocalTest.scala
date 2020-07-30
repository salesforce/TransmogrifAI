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

package com.salesforce.op.local

import java.nio.file.Paths

import com.salesforce.op.features.{Feature, FeatureLike}
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.stages.impl.classification.{BinaryClassificationModelSelector, OpLogisticRegression, OpXGBoostClassifier}
import com.salesforce.op.stages.impl.feature.StringIndexerHandleInvalid
import com.salesforce.op.stages.impl.selector.DefaultSelectorParams
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.EstimatorType
import com.salesforce.op.test.{TempDirectoryTest, TestCommon, TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomList, RandomReal, RandomText}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRow._
import com.salesforce.op.{OpWorkflow, OpWorkflowModel, UID}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class OpWorkflowModelLocalTest extends FlatSpec with TestSparkContext with TempDirectoryTest with TestCommon {
  val log = LoggerFactory.getLogger(this.getClass)
  val numRecords = 100

  // First set up the raw features
  val cityData: Seq[City] = RandomText.cities.withProbabilityOfEmpty(0.2).take(numRecords).toList
  val countryData: Seq[Country] = RandomText.countries.withProbabilityOfEmpty(0.2).take(numRecords).toList
  val pickListData: Seq[PickList] = RandomText.pickLists(domain = List("A", "B", "C", "D", "E", "F", "G", "H", "I"))
    .withProbabilityOfEmpty(0.2).limit(numRecords)
  val currencyData: Seq[Currency] = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0)
    .withProbabilityOfEmpty(0.2).limit(numRecords)

  // Generate the raw features and corresponding dataframe
  val generatedData: Seq[(City, Country, PickList, Currency)] =
    cityData.zip(countryData).zip(pickListData).zip(currencyData).map {
      case (((ci, co), pi), cu) => (ci, co, pi, cu)
    }
  val (rawDF, rawCity, rawCountry, rawPickList, rawCurrency) =
    TestFeatureBuilder("city", "country", "picklist", "currency", generatedData)

  // Construct a label with a strong signal so we make sure there's something for the algorithms to learn
  val labelSynth = new PickListLabelizer().setInput(rawPickList).getOutput().asInstanceOf[Feature[RealNN]]

  val genFeatureVector = Seq(rawCity, rawCountry, rawPickList, rawCurrency).transmogrify()
  val indexed = rawCountry.indexed(handleInvalid = StringIndexerHandleInvalid.Keep)
  val deindexed = indexed.deindexed()

  val logReg = BinaryClassificationModelSelector.Defaults.modelsAndParams.collect {
    case (lg: OpLogisticRegression, _) => lg -> new ParamGridBuilder().build()
  }
  // note: xgb needs to treat missing value as 0.0
  val xgb = BinaryClassificationModelSelector.Defaults.modelsAndParams.collect {
    case (xgb: OpXGBoostClassifier, _) => xgb ->
      new ParamGridBuilder()
        .addGrid(xgb.missing, DefaultSelectorParams.MissingValPad)
        .addGrid(xgb.objective, DefaultSelectorParams.BinaryClassXGBObjective)
        .addGrid(xgb.evalMetric, DefaultSelectorParams.BinaryClassXGBEvaluationMetric)
        .build()
  }

  lazy val (modelLocation, model, prediction) = buildAndSaveModel(logReg)
  lazy val (xgbModelLocation, xgbModel, xgbPred) = buildAndSaveModel(xgb)
  lazy val (rawData, expectedScores) = genRawDataAndScore(model, prediction)
  lazy val (rawDataXGB, expectedXGBScores) = genRawDataAndScore(xgbModel, xgbPred)
  lazy val modelLocation2 = {
    Paths.get(tempDir.toString, "op-runner-local-test-model-2").toFile.getCanonicalFile.toString
  }

  Spec(classOf[OpWorkflowModelLocal]) should "produce scores without Spark" in {
    assertLoadModelAndScore(modelLocation, rawData, expectedScores, prediction)
    assertLoadModelAndScore(xgbModelLocation, rawDataXGB, expectedXGBScores, xgbPred)
  }

  it should "produce scores without Spark in timely fashion" in {
    val scoreFn = model.scoreFunction
    scoreFn shouldBe a[ScoreFunction]
    val warmUp = rawData.map(scoreFn)
    val numOfRuns = 1000
    var elapsed = 0L
    for {_ <- 0 until numOfRuns} {
      val start = System.currentTimeMillis()
      val scores = rawData.map(scoreFn)
      elapsed += System.currentTimeMillis() - start
      assert(scores, expectedScores, prediction)
    }
    log.info(s"Scored ${expectedScores.length * numOfRuns} records in ${elapsed}ms")
    log.info(s"Average time per record: ${elapsed.toDouble / (expectedScores.length * numOfRuns)}ms")
    elapsed should be <= 10000L
  }

  it should "produce scores without Spark for all feature types" in {
    // Generate features of all possible types
    val numOfRows = 10
    val (ds, features) = TestFeatureBuilder.random(numOfRows)(
      // HashingTF transformer used in vectorization of text lists does not handle nulls well,
      // therefore setting minLen = 1 for now
      textLists = RandomList.ofTexts(RandomText.strings(0, 10), minLen = 1, maxLen = 10).limit(numOfRows)
    )
    // Prepare the label feature
    val label = features.find(_.isSubtypeOf[RealNN]).head.asInstanceOf[Feature[RealNN]].transformWith(new Labelizer)

    // Transmogrify all the features using default settings
    val featureVector = features.transmogrify()

    // Create a binary classification model selector with a single model type for simplicity
    val prediction = BinaryClassificationModelSelector.withTrainValidationSplit(
      modelsAndParameters = Seq(new OpLogisticRegression() -> new ParamGridBuilder().build())
    ).setInput(label, featureVector).getOutput()

    // Use id feature as row key
    val id = features.find(_.isSubtypeOf[ID]).head.asInstanceOf[Feature[ID]].name
    val keyFn = (r: Row) => r.getAs[String](id)
    val workflow = new OpWorkflow().setInputDataset(ds, keyFn).setResultFeatures(prediction)
    // Train, score and save the model
    val model = workflow.train()
    val expectedScoresDF = model.score()
    val expectedScores = expectedScoresDF.sort(KeyFieldName).select(prediction.name).collect().map(_.toMap)
    model.save(modelLocation2)

    // Load and score the model
    val scoreFn = OpWorkflowModel.load(modelLocation2).scoreFunction
    scoreFn shouldBe a[ScoreFunction]
    val rawData = ds.withColumn(KeyFieldName, col(id)).sort(KeyFieldName).collect().map(_.toMap)
    val scores = rawData.map(scoreFn)
    scores.length shouldBe expectedScores.length
    for {((score, expected), i) <- scores.zip(expectedScores).zipWithIndex} withClue(s"Record index $i: ") {
      score shouldBe expected
    }
  }

  private def assert(
    scores: Array[Map[String, Any]],
    expectedScores: Array[(Prediction, RealNN, RealNN, Text)],
    prediction: FeatureLike[Prediction]
  ): Unit = {
    scores.length shouldBe expectedScores.length
    for {
      ((score, (predV, labelV, indexedV, deindexedV)), i) <- scores.zip(expectedScores).zipWithIndex
      expected = Map(
        prediction.name -> predV.value,
        labelSynth.name -> labelV.value.get,
        indexed.name -> indexedV.value.get,
        deindexed.name -> deindexedV.value.orNull
      )
    } withClue(s"Record index $i: ") {
      score shouldBe expected
    }
  }

  private def assertLoadModelAndScore(
    modelLocation: String,
    rawData: Array[Map[String, Any]],
    expectedScores: Array[(Prediction, RealNN, RealNN, Text)],
    prediction: FeatureLike[Prediction]
  ): Unit = {
    val scoreFn = OpWorkflowModel.load(modelLocation).scoreFunction
    scoreFn shouldBe a[ScoreFunction]
    val scores = rawData.map(scoreFn)
    assert(scores, expectedScores, prediction)
  }

  private def buildAndSaveModel(modelsAndParams: Seq[(EstimatorType, Array[ParamMap])]):
  (String, OpWorkflowModel, FeatureLike[Prediction]) = {
    val prediction = BinaryClassificationModelSelector.withTrainValidationSplit(
      modelsAndParameters = modelsAndParams
    ).setInput(labelSynth, genFeatureVector).getOutput()
    val workflow = new OpWorkflow().setInputDataset(rawDF).setResultFeatures(prediction, labelSynth, indexed, deindexed)
    lazy val model = workflow.train()
    val path = Paths.get(tempDir.toString, "op-runner-local-test-model").toFile.getCanonicalFile.toString
    model.save(path)
    (path, model, prediction)
  }

  private def genRawDataAndScore(model: OpWorkflowModel, prediction: FeatureLike[Prediction]):
  (Array[Map[String, Any]], Array[(Prediction, RealNN, RealNN, Text)]) = {
    val rawData = rawDF.collect().map(_.toMap)
    val expectedScores = model.score().collect(prediction, labelSynth, indexed, deindexed)
    (rawData, expectedScores)
  }

}


class Labelizer(uid: String = UID[Labelizer]) extends UnaryTransformer[RealNN, RealNN]("labelizer", uid) {
  override def outputIsResponse: Boolean = true

  def transformFn: RealNN => RealNN = v => v.value.map(x => if (x > 0.0) 1.0 else 0.0).toRealNN(0.0)
}

// Label transformation function to generate a binary response from the generated picklist
class PickListLabelizer(uid: String = UID[PickListLabelizer])
  extends UnaryTransformer[PickList, RealNN]("picklistLabelizer", uid) {
  override def outputIsResponse: Boolean = true

  def transformFn: PickList => RealNN = p => p.value match {
    case Some("A") | Some("B") | Some("C") | Some("D") => RealNN(1.0)
    case _ => RealNN(0.0)
  }
}
