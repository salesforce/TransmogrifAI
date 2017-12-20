/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

import language.postfixOps
import java.io.{ByteArrayOutputStream, File, FileOutputStream, FileWriter, StringReader}

import com.salesforce.op.test.TestCommon
import org.scalatest.{Assertions, BeforeAndAfter, FlatSpec}

import scala.io.Source
import sys.process._

/**
 * Test for generator operations
 */
class CliCodeGenerationTest extends CliTestBase {

  Spec[CliExec] should "crash when running with bad avcs (the prototype one)" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", TestCsvHeadless,
      "--id", "passengerId",
      "--response", "survived",
      "--schema", "templates/simple/src/main/avro/Passenger.avsc",
      "--answers", "cli/passengers.answers",
      ProjectName,
      "--overwrite")
    result.outcome shouldBe a[Crashed]
    result.err should include ("Response field 'AInt(Survived, nullable)' cannot be nullable in ")
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
    val sut = new Sut("0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n")
    val result = sut.run(
      "gen",
      "--input", TestCsvHeadless,
      "--id", "passengerId",
      "--response", "survived",
      "--schema", TestAvsc,
      ProjectName,
      "--overwrite")
    withClue(result.err) { result.outcome shouldBe Succeeded }
  }

  Spec[CliExec] should "work with autogeneration" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", "test-data/PassengerDataWithHeader.csv",
      "--id", "passengerId",
      "--response", "survived",
      "--answers", "cli/passengers.answers",
      "--auto", "Passenger",
      ProjectName,
      "--overwrite")

    result.outcome match {
      case Succeeded => succeed
      case bad@Crashed(msg, code) =>
        bad.printStackTrace()
        fail(s"Code=$code, msg='$msg'")
        succeed
    }
  }

  Spec[CliExec] should "complain properly if neither avro on auto is specified" in {
    val sut = new Sut
    val result = sut.run(
      "gen",
      "--input", "test-data/PassengerDataWithHeader.csv",
      "--id", "passengerId",
      "--response", "survived",
      "--answers", "cli/passengers.answers",
      ProjectName,
      "--overwrite")
    result.outcome shouldBe Crashed("wrong arguments", 1)
  }
}
