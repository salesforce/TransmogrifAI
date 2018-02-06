/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

import scala.language.postfixOps

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
    assertResult(result, Succeeded)
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
