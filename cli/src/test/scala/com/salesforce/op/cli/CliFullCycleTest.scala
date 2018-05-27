/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
