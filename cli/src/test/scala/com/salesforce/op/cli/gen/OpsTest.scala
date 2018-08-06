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
