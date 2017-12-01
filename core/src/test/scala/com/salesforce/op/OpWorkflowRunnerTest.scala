/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import java.io.File

import com.salesforce.op.OpWorkflowRunType._
import com.salesforce.op.evaluators.{BinaryClassificationMetrics, Evaluators}
import com.salesforce.op.stages.impl.classification.OpLogisticRegression
import com.salesforce.op.test.PassengerSparkFixtureTest
import org.apache.commons.io.FileUtils
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.collection.JavaConverters._


@RunWith(classOf[JUnitRunner])
class OpWorkflowRunnerTest extends FlatSpec with PassengerSparkFixtureTest {

  val thisDir = new File("resources/tmp/OpWorkflowRunnerTest/")

  override def afterAll: Unit = {
    super.afterAll()
    try deleteRecursively(thisDir) finally super.afterAll()
  }

  private val features = Seq(height, weight, gender, description, age).transmogrify()
  private val survivedNum = survived.occurs()

  val (pred, raw, prob) = new OpLogisticRegression().setInput(survivedNum, features).getOutput()
  private val workflow = new OpWorkflow().setResultFeatures(pred, raw, survivedNum).setReader(dataReader)
  private val evaluator =
    Evaluators.BinaryClassification()
      .setLabelCol(survivedNum).setPredictionCol(pred).setRawPredictionCol(raw)

  val testRunner = new OpWorkflowRunner(
    workflow = workflow,
    trainingReader = dataReader,
    scoringReader = dataReader,
    evaluationReader = Some(dataReader),
    featureToComputeUpTo = gender,
    evaluator = evaluator,
    scoringEvaluator = Some(evaluator)
  )

  val invalidParamsLocation = Some(resourceFile(name = "RunnerParamsInvalid.json").getPath)
  val paramsLocation = Some(resourceFile(name = "RunnerParams.json").getPath)

  val testConfig = OpWorkflowRunnerConfig(paramLocation = paramsLocation)

  Spec[OpWorkflowRunner] should "correctly determine if the command line options are valid for each run type" in {

    val confT = OpWorkflowRunnerConfig(Train, modelLocation = Some("Test"))
    confT.validate.isRight shouldBe true

    val confS = OpWorkflowRunnerConfig(Score, modelLocation = Some("Test"), writeLocation = Some("Test"))
    confS.validate.isRight shouldBe true

    val confF = OpWorkflowRunnerConfig(Features, writeLocation = Some("Test"))
    confF.validate.isRight shouldBe true

    val confE = OpWorkflowRunnerConfig(Evaluate, modelLocation = Some("Test"), metricsLocation = Some("Test"))
    confE.validate.isRight shouldBe true

    val confTError = OpWorkflowRunnerConfig(Train, readLocations = Map("Test" -> "Test"))
    confTError.validate shouldBe Left("Must provide location to store model when training")

    val confSError = OpWorkflowRunnerConfig(Score)
    confSError.validate shouldBe Left("Must provide locations to read model and write data when scoring")

    val confFError = OpWorkflowRunnerConfig(Features)
    confFError.validate shouldBe Left("Must provide location to write data when generating features")

    val confEError = OpWorkflowRunnerConfig(Evaluate)
    confEError.validate shouldBe Left("Must provide locations to read model and write metrics when evaluating")

    val confT2 = OpWorkflowRunnerConfig(Train, paramLocation = paramsLocation)
    confT2.validate.isRight shouldBe true

    val confS2 = OpWorkflowRunnerConfig(Score, paramLocation = paramsLocation)
    confS2.validate.isRight shouldBe true

    val confF2 = OpWorkflowRunnerConfig(Features, paramLocation = paramsLocation)
    confF2.validate.isRight shouldBe true

    val confE2 = OpWorkflowRunnerConfig(Evaluate, paramLocation = paramsLocation)
    confE2.validate.isRight shouldBe true

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
    val res = doRun(runConfig, modelLocation, modelMetricsLocation)
    res shouldBe a[TrainResult]
    res.asInstanceOf[TrainResult].modelSummary shouldBe "{ }"
  }

  it should "score a dataset with a trained model" in {
    lazy val scoresLocation = new File(thisDir + "/op-runner-write/score")
    lazy val scoringMetricsLocation = new File(thisDir + "/op-runner-metrics/score")

    val runConfig = testConfig.copy(
      runType = Score,
      writeLocation = Some(scoresLocation.toString),
      metricsLocation = Some(scoringMetricsLocation.toString)
    )
    val res = doRun(runConfig, scoresLocation, scoringMetricsLocation)
    res shouldBe a[ScoreResult]
    res.asInstanceOf[ScoreResult].metrics.isDefined shouldBe true
  }

  it should "evaluate a dataset with a trained model" in {
    lazy val metricsLocation = new File(thisDir + "/op-runner-metrics/eval")

    val runConfig = testConfig.copy(
      runType = Evaluate,
      metricsLocation = Some(metricsLocation.toString)
    )
    val res = doRun(runConfig, metricsLocation)
    res shouldBe a[EvaluateResult]
    res.asInstanceOf[EvaluateResult].metrics shouldBe a[BinaryClassificationMetrics]
  }

  it should "compute features upto with a workflow" in {
    lazy val featuresLocation = new File(thisDir + "/op-runner-write/features")

    val runConfig = testConfig.copy(
      runType = Features,
      writeLocation = Some(featuresLocation.toString)
    )
    val res = doRun(runConfig, featuresLocation)
    res shouldBe a[FeaturesResult]
  }

  private def doRun(rc: OpWorkflowRunnerConfig, outFiles: File*): OpWorkflowRunnerResult = {
    val res = testRunner.run(rc.runType, rc.toOpParams.get)

    for {outFile <- outFiles} {
      outFile.exists shouldBe true
      outFile.isDirectory shouldBe true
      // TODO: maybe do a thorough files inspection here
      val files = FileUtils.listFiles(outFile, null, true)
      files.asScala.map(_.toString).exists(_.contains("_SUCCESS")) shouldBe true
      files.size > 1
    }
    res
  }

}
