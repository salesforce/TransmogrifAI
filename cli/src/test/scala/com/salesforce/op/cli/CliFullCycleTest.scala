/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

import java.io.{ByteArrayOutputStream, File, FileWriter}

import scala.language.postfixOps
import scala.sys.process._

/**
 * Test for generator operations
 */
class CliFullCycleTest extends CliTestBase {

  Spec[CliExec] should "do full cycle with avcs present" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", TestCsvHeadless,
      "--id", "passengerId",
      "--response", "survived",
      "--schema", TestAvsc,
      "--answers", "cli/passengers.answers",
      ProjectName,
      "--overwrite")
    withClue(result.err) { result.outcome shouldBe Succeeded }
    checkScalaFiles(shouldNotContain = "_code")
    checkAvroFile(new File(TestAvsc))

    runSampleWithGradle
  }

  Spec[CliExec] should "do full cycle with autoreader" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", TestCsvWithHeaders,
      "--id", "passengerId",
      "--response", "survived",
      "--auto", "Pasajeros",
      "--answers", "cli/passengers.answers",
      ProjectName,
      "--overwrite")
    withClue(result.err) { result.outcome shouldBe Succeeded }
    checkScalaFiles(shouldNotContain = "_code")

// unfortunately, it fails, due to bad data.    runSampleWithGradle
  }

  private def runSampleWithGradle = {
    val trainMe = appRuntimeArgs("train")
    val cmd = List(
      "./gradlew",
      s"sparkSubmit",
      s"-Dmain=com.salesforce.app.$ProjectName",
      s"""-Dargs=\"$trainMe\"""")
    val cmdToRunManuallyInYourConsole = cmd mkString " "
    val shell = new File(projectDir, "train")
    val shellW = new FileWriter(shell)
    shellW.write(cmdToRunManuallyInYourConsole)
    shellW.close()

    val proc = Process("sh" :: "train" :: Nil, projectDir.getAbsoluteFile)
    val stdOut = new ByteArrayOutputStream
    val stdErr = new ByteArrayOutputStream
    val logger = ProcessLogger(stdout.print, stderr.print)
    val code = proc !< logger

    withClue(s"code=$code\nerr=${stdErr.toString}\nout=${stdOut.toString}") {
      code shouldBe 0
    }
  }
}
