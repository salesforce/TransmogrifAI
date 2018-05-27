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

import java.io.File
import java.nio.file.{Files, Path, Paths}

import com.salesforce.op.cli.gen.{Ops, ProblemSchema}

import scala.util.{Failure, Success, Try}

/**
 * Represents a builder for the command to generate a project.
 *
 * @param projName   The name of the project (in CamelCase)
 * @param inputFile  The file to use for input (should be csv)
 * @param response   The name of the response field
 * @param idField    The name of the ID field
 * @param schemaSource The way data schema is discovered
 * @param answersFile The file with answers to questions; using stdin if missing
 * @param overwrite tells whether we can override the existing project folder
 */
case class CliParameters
(
  command: String = "???",
  location: File = new File("."),
  projName: String = "Sample",
  inputFile: Option[File] = None,
  response: Option[String] = None,
  idField: Option[String] = None,
  schemaSource: Option[SchemaSource] = None,
  answersFile: Option[File] = None,
  overwrite: Boolean = false
) {

  private def delete(f: File): Unit = {
    Option(f.listFiles).foreach(_ foreach delete)
    f.delete()

    if (f.exists()) {
      throw new IllegalStateException(s"Directory '${f.getAbsolutePath}' still exists")
    }
  }

  private def ensureProjectDir(projectName: String): Option[File] = Try {
    val dir = new File(location, projName.toLowerCase)
    if (overwrite) delete(dir)

    dir.mkdirs()
    if (!dir.exists()) {
      throw new IllegalStateException(s"Failed to create directory '${dir.getAbsolutePath}'")
    }
    dir
  } match {
    case Success(f) => Some(f)
    case Failure(t) =>
      println(t.getMessage)
      None
  }

  def values: Option[GeneratorConfig] = for {
    inf <- inputFile
    resp <- response
    idf <- idField
    sd <- schemaSource
    pd <- ensureProjectDir(projName)
  } yield GeneratorConfig(
    projName = projName,
    projectDirectory = pd,
    inputFile = inf,
    response = resp,
    idField = idf,
    schemaSource = sd,
    answersFile = answersFile)
}

/**
 * Represents a command to generate a project.
 *
 * @param projName   The name of the project (in CamelCase)
 * @param projectDirectory The directory to generate a new project in.
 * @param inputFile  The file to use for input (should be csv)
 * @param response   The name of the response field
 * @param idField    The name of the ID field
 * @param schemaSource The way data schema is discovered
 * @param answersFile The file with answers to questions; using stdin if missing
 */
case class GeneratorConfig
(
  command: String = "",
  projName: String,
  projectDirectory: File,
  inputFile: File,
  response: String,
  idField: String,
  schemaSource: SchemaSource,
  answersFile: Option[File]
)
