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

import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.stages.impl.classification.{BinaryClassificationModelSelector, OpLogisticRegression}
import com.salesforce.op.stages.impl.feature.StringIndexerHandleInvalid
import com.salesforce.op.test.{PassengerSparkFixtureTest, TestCommon, TestFeatureBuilder}
import com.salesforce.op.testkit.{RandomList, RandomText}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRow._
import com.salesforce.op.{OpParams, OpWorkflow, OpWorkflowModel, UID}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class OpWorkflowRunnerLocalTest extends FlatSpec with PassengerSparkFixtureTest with TestCommon {

  val log = LoggerFactory.getLogger(this.getClass)

  val features = Seq(height, weight, gender, description, age).transmogrify()
  val survivedNum = survived.occurs()

  val indexed = description.indexed(handleInvalid = StringIndexerHandleInvalid.Keep)
  val deindexed = indexed.deindexed()

  val logReg = BinaryClassificationModelSelector.Defaults.modelsAndParams.collect {
    case (lg: OpLogisticRegression, _) => lg -> new ParamGridBuilder().build()
  }

  val prediction = BinaryClassificationModelSelector.withTrainValidationSplit(
    modelsAndParameters = logReg, splitter = None
  ).setInput(survivedNum, features).getOutput()

  val workflow = new OpWorkflow().setReader(dataReader)
    .setResultFeatures(prediction, survivedNum, indexed, deindexed)

  lazy val model = workflow.train()
  lazy val modelLocation = {
    val path = Paths.get(tempDir.toString, "op-runner-local-test-model").toFile.getCanonicalFile.toString
    model.save(path)
    path
  }
  lazy val rawData = dataReader.generateDataFrame(model.getRawFeatures()).sort(KeyFieldName).collect().map(_.toMap)
  lazy val expectedScores = model.score().sort(KeyFieldName).collect(prediction, survivedNum, indexed, deindexed)
  lazy val modelLocation2 = {
    Paths.get(tempDir.toString, "op-runner-local-test-model-2").toFile.getCanonicalFile.toString
  }

  Spec(classOf[OpWorkflowRunnerLocal]) should "produce scores without Spark" in {
    val params = new OpParams().withValues(modelLocation = Some(modelLocation))
    val scoreFn = new OpWorkflowRunnerLocal(workflow).scoreFunction(params)
    scoreFn shouldBe a[ScoreFunction]
    val scores = rawData.map(scoreFn)
    assert(scores, expectedScores)
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
      assert(scores, expectedScores)
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
    expectedScores: Array[(Prediction, RealNN, RealNN, Text)]
  ): Unit = {
    scores.length shouldBe expectedScores.length
    for {
      ((score, (predV, survivedV, indexedV, deindexedV)), i) <- scores.zip(expectedScores).zipWithIndex
      expected = Map(
        prediction.name -> predV.value,
        survivedNum.name -> survivedV.value.get,
        indexed.name -> indexedV.value.get,
        deindexed.name -> deindexedV.value.orNull
      )
    } withClue(s"Record index $i: ") {
      score shouldBe expected
    }
  }

}


class Labelizer(uid: String = UID[Labelizer]) extends UnaryTransformer[RealNN, RealNN]("labelizer", uid) {
  override def outputIsResponse: Boolean = true
  def transformFn: RealNN => RealNN = v => v.value.map(x => if (x > 0.0) 1.0 else 0.0).toRealNN(0.0)
}
