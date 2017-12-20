/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
