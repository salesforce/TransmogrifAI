/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

import language.postfixOps
import java.io.File

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

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

  it should "read answers from a file" in {
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
    result.err should include("Bad data file")
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
