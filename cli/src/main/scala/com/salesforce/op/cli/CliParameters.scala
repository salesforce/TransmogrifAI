/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

import java.io.File
import java.nio.file.{Files, Path, Paths}

import com.salesforce.op.cli.gen.{Ops, ProblemSchema}

import scala.util.{Failure, Success, Try}

case class CliParameters
(
  command: String = "",
  settings: GeneratorConfigurator = GeneratorConfigurator()
)

/**
 * Represents a builder for the command to generate a project.
 *
 * @param projName   The name of the project (in CamelCase)
 * @param inputFile  The file to use for input (should be csv)
 * @param response   The name of the response field
 * @param idField    The name of the ID field
 * @param schemaFile The file to read a schema from
 * @param answersFile The file with answers to questions; using stdin if missing
 * @param overrideit tells whether we can override the existing project folder
 */
case class GeneratorConfigurator
(
  location: File = new File("."),
  projName: String = "Sample",
  inputFile: Option[File] = None,
  response: Option[String] = None,
  idField: Option[String] = None,
  schemaFile: Option[File] = None,
  answersFile: Option[File] = None,
  overrideit: Boolean = false
) {

  private def delete(dir: File): Unit = {
    val contents = Option(dir.listFiles).map(_.toList).toList.flatten

    for {
      f <- contents
    } {
      delete(f)
    }
    dir.delete()

    if (dir.exists()) {
      throw new IllegalStateException(s"Directory '${dir.getAbsolutePath}' still exists")
    }
  }

  private def ensureProjectDir(projectName: String): Option[File] = Try {
    val dir = new File(location, projName.toLowerCase)
    if (overrideit) delete(dir)

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
    sf <- schemaFile
    pd <- ensureProjectDir(projName)
  } yield GeneratorConfig(
    projName = projName,
    projectDirectory = pd,
    inputFile = inf,
    response = resp,
    idField = idf,
    schemaFile = sf,
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
 * @param schemaFile The file to read a schema from
 * @param answersFile The file with answers to questions; using stdin if missing
 */
case class GeneratorConfig
(
  projName: String,
  projectDirectory: File,
  inputFile: File,
  response: String,
  idField: String,
  schemaFile: File,
  answersFile: Option[File]
)
