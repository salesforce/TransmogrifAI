/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import java.io.File

import com.salesforce.op.cli.{AvroSchemaFromFile, CliParameters, GeneratorConfig}
import com.salesforce.op.test.TestCommon
import org.scalatest.{Assertions, FlatSpec}

import scala.io.Source

/**
 * Test for generator operations
 */
class OpsTest extends FlatSpec with TestCommon with Assertions {

  val tempFolder = new File(System.getProperty("java.io.tmpdir"))
  val projectFolder = new File(tempFolder, "cli_test")
  projectFolder.deleteOnExit()

  val testParams = CliParameters(
    location = tempFolder,
    projName = "cli_test",
    inputFile = Some(new File("templates/simple/src/main/resources/PassengerData.csv")),
    response = Some("survived"),
    idField = Some("passengerId"),
    schemaSource = Some(AvroSchemaFromFile(new File("utils/src/main/avro/PassengerCSV.avsc"))),
    answersFile = Some(new File("cli/passengers.answers")),
    overwrite = true).values

  Spec[Ops] should "generate project files" in {

    testParams match {
      case None =>
        fail("Could not create config, I wonder why")
      case Some(conf: GeneratorConfig) =>
        val ops = Ops(conf)
        ops.run()
        val buildFile = new File(projectFolder, "build.gradle")
        buildFile should exist

        val buildFileContent = Source.fromFile(buildFile).mkString

        buildFileContent should include("credentials artifactoryCredentials")

        val scalaSourcesFolder = new File(projectFolder, "src/main/scala/com/salesforce/app")
        val featuresFile = Source.fromFile(new File(scalaSourcesFolder, "Features.scala")).getLines
        val heightLine = featuresFile.find(_ contains "height") map (_.trim)
        heightLine shouldBe Some(
          "val height = FeatureBuilder.Real[PassengerCSV].extract(o => o.getHeight.toReal).asPredictor"
        )
    }

  }

}
