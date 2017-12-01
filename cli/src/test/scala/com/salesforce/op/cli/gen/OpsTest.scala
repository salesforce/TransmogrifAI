/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import java.io.File
import java.io.File.TempDirectory

import com.salesforce.op.cli.{GeneratorConfig, GeneratorConfigurator}
import com.salesforce.op.test.TestCommon
import org.junit.{Rule, Test}
import org.junit.rules.TemporaryFolder
import org.scalatest.{Assertions, FlatSpec}

import scala.io.Source

/**
 * Test for generator operations
 */
class OpsTest extends FlatSpec with TestCommon with Assertions {

  Spec[Ops] should "generate project files" in {
    val tempFolder = new File(System.getProperty("java.io.tmpdir"))
    val projectFolder = new File(tempFolder, "cli_test")
    projectFolder.deleteOnExit()
    projectFolder.delete()

    val testConfig = GeneratorConfigurator(
      location = tempFolder,
      projName = "cli_test",
      inputFile = Some(new File("templates/simple/src/main/resources/PassengerData.csv")),
      response = Some("survived"),
      idField = Some("passengerId"),
      schemaFile = Some(new File("utils/src/main/avro/PassengerCSV.avsc")),
      answersFile = Some(new File("cli/passengers.answers")),
      overrideit = true).values

    testConfig match {
      case None =>
        fail("Could not create config, I wonder why")
      case Some(conf) =>
        val ops = Ops(conf)
        ops.run()
        val buildFile = new File(projectFolder, "build.gradle")
        buildFile should exist

        val buildFileContent = Source.fromFile(buildFile).mkString

        buildFileContent should include ("credentials artifactoryCredentials")
    }

  }

}
