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

import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry._
import com.salesforce.op.stages.impl.feature.StringIndexerHandleInvalid
import com.salesforce.op.test.{PassengerSparkFixtureTest, TestCommon}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRow._
import com.salesforce.op.{OpParams, OpWorkflow}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class OpWorkflowRunnerLocalTest extends FlatSpec with PassengerSparkFixtureTest with TestCommon {

  val log = LoggerFactory.getLogger(this.getClass)

  val features = Seq(height, weight, gender, description, age).transmogrify()
  val survivedNum = survived.occurs()

  // TODO: remove .map[Text] once Aardpfark supports null inputs for StringIndexer
  val indexed = description.map[Text](v => if (v.isEmpty) Text("") else v)
    .indexed(handleInvalid = StringIndexerHandleInvalid.Skip)

  val prediction = BinaryClassificationModelSelector.withTrainValidationSplit(
    splitter = None, modelTypesToUse = Seq(OpLogisticRegression)
  ).setInput(survivedNum, features).getOutput()

  val workflow = new OpWorkflow().setResultFeatures(prediction, survivedNum, indexed).setReader(dataReader)

  lazy val model = workflow.train()

  lazy val modelLocation = {
    val path = Paths.get(tempDir.toString, "op-runner-local-test-model").toFile.getCanonicalFile.toString
    model.save(path)
    path
  }

  lazy val rawData = dataReader.generateDataFrame(model.rawFeatures).sort(KeyFieldName).collect().map(_.toMap)

  lazy val expectedScores = model.score().sort(KeyFieldName).collect(prediction, survivedNum, indexed)

  Spec(classOf[OpWorkflowRunnerLocal]) should "produce scores without Spark" in {
    val params = new OpParams().withValues(modelLocation = Some(modelLocation))
    val scoreFn = new OpWorkflowRunnerLocal(workflow).score(params)
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

  private def assert(
    scores: Array[Map[String, Any]],
    expectedScores: Array[(Prediction, RealNN, RealNN)]
  ): Unit = {
    scores.length shouldBe expectedScores.length
    for {
      (score, (predV, survivedV, indexedV)) <- scores.zip(expectedScores)
      expected = Map(
        prediction.name -> predV.value,
        survivedNum.name -> survivedV.value.get,
        indexed.name -> indexedV.value.get
      )
    } score shouldBe expected
  }

}
