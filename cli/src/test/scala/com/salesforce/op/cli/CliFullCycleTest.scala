/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

import java.io.{File, FileWriter}

import com.salesforce.op.OpWorkflowRunType
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.language.postfixOps
import scala.sys.process._

/**
 * End to end test: gen, build and spark submit
 */
@RunWith(classOf[JUnitRunner])
class CliFullCycleTest extends CliTestBase {

  Spec[CliExec] should "do full cycle with avcs present" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", TestCsvHeadless,
      "--id", "passengerId",
      "--response", "survived",
      "--schema", TestAvsc,
      "--answers", AnswersFile,
      ProjectName,
      "--overwrite"
    )
    assertResult(result, Succeeded)
    checkScalaFiles(shouldNotContain = "_code")
    checkAvroFile(new File(TestAvsc))
    // runTraining() // TODO: requires proper SPARK_HOME setup on TC

    // TODO: score & evaluate
  }

  it should "do full cycle with autoreader" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", TestBigCsvWithHeaders,
      "--id", "passengerId",
      "--response", "survived",
      "--auto", "Pasajeros",
      "--answers", AnswersFile,
      ProjectName,
      "--overwrite"
    )
    assertResult(result, Succeeded)
    checkScalaFiles(shouldNotContain = "_code")
    // TODO: unfortunately, it fails, due to bad data.
    // runTraining() // TODO: requires proper SPARK_HOME setup on TC

    // TODO: score & evaluate
  }

  // TODO: add tests for multiclass & regression models


  private def runBuild() = runCommand(List("./gradlew", "--no-daemon", "installDist"))

  private def runTraining() = {
    val trainMe = appRuntimeArgs(OpWorkflowRunType.Train)
    val cmd = List(
      "./gradlew",
      "--no-daemon",
      s"sparkSubmit",
      s"-Dmain=com.salesforce.app.$ProjectName",
      s"""-Dargs=\"$trainMe\""""
    )
    runCommand(cmd)
  }

  private def runCommand(cmd: List[String]) = {
    val cmdStr = cmd.mkString(" ")
    val cmdSh = new FileWriter(new File(projectDir, "cmd"))
    cmdSh.write(cmdStr)
    cmdSh.close()

    val proc = Process("sh" :: "cmd" :: Nil, projectDir.getAbsoluteFile)
    val logger = ProcessLogger(s => log.info(s), s => log.error(s))
    val code = proc !< logger

    if (code == 0) succeed else fail(s"Command returned a non zero code: $code")
  }
}
