/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import java.io.File

import com.salesforce.op.OpWorkflowRunType._
import com.salesforce.op.evaluators.{BinaryClassificationMetrics, Evaluators}
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataFrameFieldNames._
import com.salesforce.op.stages.impl.classification.OpLogisticRegression
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

  val thisDir = new File("resources/tmp/OpWorkflowRunnerTest/").getCanonicalFile

  override def beforeAll: Unit = try deleteRecursively(thisDir) finally super.beforeAll
  override def afterAll: Unit = try deleteRecursively(thisDir) finally super.afterAll

  private val features = Seq(height, weight, gender, description, age).transmogrify()
  private val survivedNum = survived.occurs()

  val (pred, raw, prob) = new OpLogisticRegression().setInput(survivedNum, features).getOutput()
  private val workflow = new OpWorkflow().setResultFeatures(pred, raw, survivedNum).setReader(dataReader)
  private val evaluator =
    Evaluators.BinaryClassification().setLabelCol(survivedNum).setPredictionCol(pred).setRawPredictionCol(raw)

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
    lazy val modelLocation = new File(thisDir + "/op-runner-test-model")
    lazy val modelMetricsLocation = new File(thisDir + "/op-runner-test-metrics/train")

    val runConfig = testConfig.copy(
      runType = Train,
      modelLocation = Some(modelLocation.toString),
      metricsLocation = Some(modelMetricsLocation.toString)
    )
    val res = doRun[TrainResult](runConfig, modelLocation, modelMetricsLocation)
    res.modelSummary shouldBe "{ }"
  }

  it should "score a dataset with a trained model" in {
    val scoresLocation = new File(thisDir + "/op-runner-test-write/score")
    val scoringMetricsLocation = new File(thisDir + "/op-runner-test-metrics/score")

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
    val readLocation = new File(thisDir + "/op-runner-test-read/streaming-score")
    val scoresLocation = new File(thisDir + "/op-runner-test-write/streaming-score")

    // Prepare streaming input data
    FileUtils.forceMkdir(readLocation)
    val passengerAvroFile = new File(passengerAvroPath).getCanonicalFile
    FileUtils.copyFile(passengerAvroFile, new File(readLocation + "/" + passengerAvroFile.getName), false)

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
    val metricsLocation = new File(thisDir + "/op-runner-test-metrics/eval")

    val runConfig = testConfig.copy(
      runType = Evaluate,
      metricsLocation = Some(metricsLocation.toString)
    )
    val res = doRun[EvaluateResult](runConfig, metricsLocation)
    res.metrics shouldBe a[BinaryClassificationMetrics]
  }

  it should "compute features upto with a workflow" in {
    lazy val featuresLocation = new File(thisDir + "/op-runner-test-write/features")

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
