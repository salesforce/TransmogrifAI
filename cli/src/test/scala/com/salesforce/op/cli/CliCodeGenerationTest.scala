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

package com.salesforce.op.cli

import language.postfixOps
import java.io.File
import java.nio.file.Paths

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.io.Source

/**
 * Test for generator operations
 */
@RunWith(classOf[JUnitRunner])
class CliCodeGenerationTest extends CliTestBase {

  Spec[CliExec] should "crash when running with bad avcs (the prototype one)" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", TestCsvHeadless,
      "--id", "passengerId",
      "--response", "survived",
      "--schema", AvcsSchema,
      "--answers", AnswersFile,
      ProjectName,
      "--overwrite"
    )
    result.outcome shouldBe a[Crashed]
    result.err should include("Response field 'AInt(Survived, nullable)' cannot be nullable in ")
  }

  it should "explain the need of having answers file available" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", TestCsvHeadless,
      "--id", "passengerId",
      "--response", "survived",
      "--schema", TestAvsc,
      "--answers", "/home/alone",
      ProjectName,
      "--overwrite"
    )

    result.outcome shouldBe Crashed("wrong arguments", 1)
    result.err should include("Error: File '/home/alone' not found")
  }

  it should "talk to the user if answers file is missing" in {
    val sut = new Sut("0\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n")
    val result = sut.run(
      "gen",
      "--input", TestCsvHeadless,
      "--id", "passengerId",
      "--response", "survived",
      "--schema", TestAvsc,
      ProjectName,
      "--overwrite"
    )
    assertResult(result, Succeeded)
  }

  it should "generage code, answers in a file" in {
    val sut = new Sut()
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
    withClue(result.err) {
      result.outcome shouldBe Succeeded
    }
  }

  it should "not fail when fields have underscores in names" in {
    val sut = new Sut()
    val result = sut.run(
      "gen",
      "--input", TestCsvHeadless,
      "--id", "passengerId",
      "--response", "survived",
      "--schema", TestAvscWithUnderscores,
      "--answers", AnswersFileWithUnderscores,
      ProjectName,
      "--overwrite"
    )
    withClue(result.err) {
      result.outcome shouldBe Succeeded
    }
    val scalaSourcesFolder = Paths.get(projectFolder, "src", "main", "scala", "com", "salesforce", "app")

    val featuresFile = Source.fromFile(new File(scalaSourcesFolder.toFile, "Features.scala")).getLines
    val testLines = featuresFile.dropWhile(!_.contains("val p_class = FB"))
    testLines.hasNext shouldBe true
    testLines.next
    val integralPassenger = testLines.next
    integralPassenger.trim shouldBe ".Integral[Passenger]"
    val thisOneShouldNotHaveUnderscoreInGetter = testLines.next
    thisOneShouldNotHaveUnderscoreInGetter.trim shouldBe ".extract(_.getPClass.toIntegral)"
  }

  it should "work with autogeneration" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", TestSmallCsvWithHeaders,
      "--id", "passengerId",
      "--response", "survived",
      "--answers", AnswersFile,
      "--auto", "Passenger",
      ProjectName,
      "--overwrite"
    )
    assertResult(result, Succeeded)

    val folder = new File(ProjectName.toLowerCase)
    folder.exists() shouldBe true

    val expectedContent =
      Set("README.md", "gradle", "gradlew", "spark.gradle", ".gitignore",
        "build.gradle", "gradle.properties", "settings.gradle", "src")

    val content = folder.list()
    content.toSet shouldBe expectedContent

  }

  it should "react properly on missing headers in data file" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", findFile("test-data/PassengerDataAll.csv"),
      "--id", "passengerId",
      "--response", "survived",
      "--auto", "Passenger",
      ProjectName,
      "--overwrite")

    result.outcome shouldBe a[Crashed]
    withClue(result.err) {
      result.err should include("Bad data file: " + result.err)
    }
    val folder = new File(ProjectName.toLowerCase)
    folder.exists() shouldBe false

  }

  it should "complain properly if neither avro nor auto is specified" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", TestSmallCsvWithHeaders,
      "--id", "passengerId",
      "--response", "survived",
      "--answers", AnswersFile,
      ProjectName,
      "--overwrite"
    )
    assertResult(result, Crashed("wrong arguments", 1))
  }

}
