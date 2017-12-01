/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli

import java.io.File

trait OpCli {
  self: scopt.OptionParser[CliParameters] =>

  private val num = ('0' to '9').toSet
  private val alphaNum = (('a' to 'z') ++ ('A' to 'Z') ++ num).toSet

  def camelCase(name: String): Either[String, Unit] = {
    if (num(name.charAt(0))) failure("First character of name should be letter (name should be CamelCase)")
    else if (!name.charAt(0).isUpper) failure("First letter of name should be uppercase (name should be CamelCase)")
    else if (name.contains('-')) failure("Name should be camelCase, not Kebab-case")
    else if (name.contains('_')) failure("Name should be camelCase, not Underscore_case")
    else if (!name.forall(alphaNum)) failure("Name should only contain alphanumeric (name should be CamelCase)")
    else success
  }

  def fileExists(file: File): Either[String, Unit] = {
    if (!file.exists()) failure(s"File '${file.getAbsolutePath}' not found")
    else success
  }
}

object CommandParser extends scopt.OptionParser[CliParameters]("op") with OpCli {
  head("op: Optimus Prime CLI")

  help("help")

  note("")
  cmd("gen")
    .action((_, cfg) => cfg.copy(command = "gen"))
    .text("Generate a new Optimus Prime project")
    .children(
      arg[String]("name")
        .text("Name for the Optimus Prime project (should be in CamelCase) [required]")
        .validate(camelCase)
        .action((appName, cfg) => cfg.copy(settings = cfg.settings.copy(projName = appName))),

      opt[File]("input")
        .text("Input file for the Optimus Prime project [required]")
        .validate(fileExists)
        .required
        .action((inputFile, cfg) => cfg.copy(settings = cfg.settings.copy(inputFile = Option(inputFile)))),

      opt[String]("id")
        .text("Name for the ID field [required]")
        .required
        .action((id, cfg) => cfg.copy(settings = cfg.settings.copy(idField = Option(id)))),

      opt[String]("response")
        .text("Feature name to predict [required]")
        .required
        .action((response, cfg) => cfg.copy(settings = cfg.settings.copy(response = Option(response)))),

      opt[File]("answers")
        .text("answers to generator questions [optional]")
        .validate(fileExists)
        .required
        .action((answersFile, cfg) => cfg.copy(settings = cfg.settings.copy(answersFile = Option(answersFile)))),

      opt[Unit]("override")
        .action( (_, cfg) => cfg.copy(settings = cfg.settings.copy(overrideit = true))),

      opt[File]("schema")
        .text("Avro schema to use for objects [required]")
        .validate(fileExists)
        .required
        .action((schemaFile, cfg) => cfg.copy(settings = cfg.settings.copy(schemaFile = Option(schemaFile))))
    )

}
