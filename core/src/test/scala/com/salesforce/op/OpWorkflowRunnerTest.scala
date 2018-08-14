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

package com.salesforce.op

import java.io.File
import java.nio.file.Paths

import com.salesforce.op.OpWorkflowRunType._
import com.salesforce.op.evaluators.{BinaryClassificationMetrics, Evaluators}
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry.LogisticRegression
import com.salesforce.op.test.{PassengerSparkFixtureTest, TestSparkStreamingContext}
import com.salesforce.op.utils.spark.AppMetrics
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.commons.io.FileUtils
import org.junit.runner.RunWith
import org.scalactic.source
import org.scalatest.AsyncFlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.concurrent.Promise
import scala.reflect.ClassTag


@RunWith(classOf[JUnitRunner])
class OpWorkflowRunnerTest extends AsyncFlatSpec
  with PassengerSparkFixtureTest with TestSparkStreamingContext {

  val log = LoggerFactory.getLogger(this.getClass)

  val thisDir = Paths.get("resources", "tmp", "OpWorkflowRunnerTest").toFile.getCanonicalFile

  override def beforeAll: Unit = try deleteRecursively(thisDir) finally super.beforeAll
  override def afterAll: Unit = try deleteRecursively(thisDir) finally super.afterAll

  private val features = Seq(height, weight, gender, description, age).transmogrify()
  private val survivedNum = survived.occurs()

  val (pred, raw, prob) = BinaryClassificationModelSelector.withTrainValidationSplit(None)
    .setModelsToTry(LogisticRegression)
    .setLogisticRegressionRegParam(0)
    .setInput(survivedNum, features).getOutput()
  private val workflow = new OpWorkflow().setResultFeatures(pred, raw, survivedNum).setReader(dataReader)
  private val evaluator =
    Evaluators.BinaryClassification().setLabelCol(survivedNum).setPredictionCol(pred).setRawPredictionCol(raw)
      .setProbabilityCol(prob)

  val metricsPromise = Promise[AppMetrics]()

  val testRunner = new OpWorkflowRunner(
    workflow = workflow,
    trainingReader = simpleReader,
    scoringReader = simpleReader,
    evaluationReader = Some(simpleReader),
    streamingScoreReader = Some(simpleStreamingReader),
    featureToComputeUpTo = Some(gender),
    evaluator = Some(evaluator),
    scoringEvaluator = Some(evaluator)
  ).addApplicationEndHandler(collectMetrics(metricsPromise))

  val invalidParamsLocation = Some(resourceFile(name = "RunnerParamsInvalid.json").getPath)
  val paramsLocation = Some(resourceFile(name = "RunnerParams.json").getPath)
  val testConfig = OpWorkflowRunnerConfig(paramLocation = paramsLocation)

  Spec[OpWorkflowRunner] should "correctly determine if the command line options are valid for each run type" in {
    assertConf(OpWorkflowRunnerConfig(Train, modelLocation = Some("Test")))
    assertConf(OpWorkflowRunnerConfig(Score, modelLocation = Some("Test"), writeLocation = Some("Test")))
    assertConf(OpWorkflowRunnerConfig(StreamingScore, modelLocation = Some("Test"), writeLocation = Some("Test")))
    assertConf(OpWorkflowRunnerConfig(Features, writeLocation = Some("Test")))
    assertConf(OpWorkflowRunnerConfig(Evaluate, modelLocation = Some("Test"), metricsLocation = Some("Test")))

    val confTError = OpWorkflowRunnerConfig(Train, readLocations = Map("Test" -> "Test"))
    confTError.validate shouldBe Left("Must provide location to store model when training")

    val confSError = OpWorkflowRunnerConfig(Score)
    confSError.validate shouldBe Left("Must provide locations to read model and write data when scoring")

    val confFError = OpWorkflowRunnerConfig(Features)
    confFError.validate shouldBe Left("Must provide location to write data when generating features")

    val confEError = OpWorkflowRunnerConfig(Evaluate)
    confEError.validate shouldBe Left("Must provide locations to read model and write metrics when evaluating")

    assertConf(OpWorkflowRunnerConfig(Train, paramLocation = paramsLocation))
    assertConf(OpWorkflowRunnerConfig(Score, paramLocation = paramsLocation))
    assertConf(OpWorkflowRunnerConfig(StreamingScore, paramLocation = paramsLocation))
    assertConf(OpWorkflowRunnerConfig(Features, paramLocation = paramsLocation))
    assertConf(OpWorkflowRunnerConfig(Evaluate, paramLocation = paramsLocation))

    val confTError2 = OpWorkflowRunnerConfig(Train, paramLocation = invalidParamsLocation)
    confTError2.validate shouldBe Left("Must provide location to store model when training")

    val confSError2 = OpWorkflowRunnerConfig(Score, paramLocation = invalidParamsLocation)
    confSError2.validate shouldBe Left("Must provide locations to read model and write data when scoring")

    val confFError2 = OpWorkflowRunnerConfig(Features, paramLocation = invalidParamsLocation)
    confFError2.validate shouldBe Left("Must provide location to write data when generating features")

    val confEError2 = OpWorkflowRunnerConfig(Evaluate, paramLocation = invalidParamsLocation)
    confEError2.validate shouldBe Left("Must provide locations to read model and write metrics when evaluating")
  }

  it should "train a workflow and write the trained model" in {
    lazy val modelLocation = new File(thisDir, "op-runner-test-model")
    lazy val modelMetricsLocation = Paths.get(thisDir.toString, "op-runner-test-metrics", "train").toFile

    val runConfig = testConfig.copy(
      runType = Train,
      modelLocation = Some(modelLocation.toString),
      metricsLocation = Some(modelMetricsLocation.toString)
    )
    val res = doRun[TrainResult](runConfig, modelLocation, modelMetricsLocation)
    res.modelSummary.nonEmpty shouldBe true
  }

  it should "score a dataset with a trained model" in {
    val scoresLocation = Paths.get(thisDir.toString, "op-runner-test-write", "score").toFile
    val scoringMetricsLocation = Paths.get(thisDir.toString, "op-runner-test-metrics", "score").toFile

    val runConfig = testConfig.copy(
      runType = Score,
      writeLocation = Some(scoresLocation.toString),
      metricsLocation = Some(scoringMetricsLocation.toString)
    )
    val res = doRun[ScoreResult](runConfig, scoresLocation, scoringMetricsLocation)
    res.asInstanceOf[ScoreResult].metrics.isDefined shouldBe true

    val scores = loadAvro(scoresLocation.toString)
    scores.sort(KeyFieldName, pred.name).collect(pred) shouldBe Seq(0, 1, 1, 0, 0, 1, 0, 1).map(_.toRealNN)
  }

  it should "streaming score a dataset with a trained model" in {
    val readLocation = Paths.get(thisDir.toString, "op-runner-test-read", "streaming-score").toFile
    val scoresLocation = Paths.get(thisDir.toString, "op-runner-test-write", "streaming-score").toFile

    // Prepare streaming input data
    FileUtils.forceMkdir(readLocation)
    val passengerAvroFile = new File(passengerAvroPath).getCanonicalFile
    FileUtils.copyFile(passengerAvroFile, new File(readLocation, passengerAvroFile.getName), false)

    val runConfig = testConfig.copy(
      runType = StreamingScore,
      writeLocation = Some(scoresLocation.toString),
      readLocations = Map(simpleStreamingReader.typeName -> readLocation.toString)
    )
    val res = doRun[StreamingScoreResult](runConfig)

    val scoresDirs = scoresLocation.listFiles().filter(_.isDirectory)
    scoresDirs.length shouldBe 1

    val scores = loadAvro(scoresDirs.head.toString)
    scores.sort(KeyFieldName, pred.name).collect(pred) shouldBe Seq(0, 1, 1, 0, 0, 1, 0, 1).map(_.toRealNN)
  }

  it should "evaluate a dataset with a trained model" in {
    val metricsLocation = Paths.get(thisDir.toString, "op-runner-test-metrics", "eval").toFile

    val runConfig = testConfig.copy(
      runType = Evaluate,
      metricsLocation = Some(metricsLocation.toString)
    )
    val res = doRun[EvaluateResult](runConfig, metricsLocation)
    res.metrics shouldBe a[BinaryClassificationMetrics]
  }

  it should "compute features upto with a workflow" in {
    lazy val featuresLocation = Paths.get(thisDir.toString, "op-runner-test-write", "features").toFile

    val runConfig = testConfig.copy(
      runType = Features,
      writeLocation = Some(featuresLocation.toString)
    )
    val res = doRun[FeaturesResult](runConfig, featuresLocation)
    res shouldBe a[FeaturesResult]
  }

  it should "collect and report metrics on application end" in {
    spark.stop()
    metricsPromise.future.map { metrics =>
      metrics.appId.isEmpty shouldBe false
      OpWorkflowRunType.withNameInsensitiveOption(metrics.runType).isDefined shouldBe true
      metrics.appName shouldBe "op-test"
      metrics.appStartTime should be >= 0L
      metrics.appEndTime should be >= 0L
      metrics.appDuration should be >= 0L
      metrics.stageMetrics.length should be > 0
    }
  }

  private def assertConf(c: OpWorkflowRunnerConfig)(implicit pos: source.Position) = {
    val res = c.validate
    if (res.isLeft) fail(message = res.left.get)(pos)
  }

  private def doRun[R <: OpWorkflowRunnerResult: ClassTag](rc: OpWorkflowRunnerConfig, outFiles: File*): R = {
    log.info("OpWorkflowRunnerConfig:\n{}", rc.toString)
    val res = testRunner.run(rc.runType, rc.toOpParams.get)
    for {outFile <- outFiles} {
      outFile.exists shouldBe true
      outFile.isDirectory shouldBe true
      // TODO: maybe do a thorough files inspection here
      val files = FileUtils.listFiles(outFile, null, true)
      files.asScala.map(_.toString).exists(_.contains("_SUCCESS")) shouldBe true
      files.size > 1
    }
    res shouldBe a[R]
    res.asInstanceOf[R]
  }

  private def collectMetrics(p: Promise[AppMetrics])(appMetrics: AppMetrics): Unit = {
    log.info("App metrics:\n{}", appMetrics)
    if (!p.isCompleted) p.success(appMetrics)
  }

}
