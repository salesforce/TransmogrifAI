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

  private val Identifier = "([a-zA-Z]\\w*)".r

  def isIdentifier(name: String): Either[String, Unit] = {
    name match {
      case Identifier(_) => success
      case bad => failure(s"Expected an identifier, got '$name'")
    }
  }
}

object CommandParser extends scopt.OptionParser[CliParameters]("transmogrifai") with OpCli {
  private[cli] var AUTO_ENABLED = true // for testing, temporary

  head("transmogrifai: TransmogrifAI CLI")

  help("help")

  note("")

  val knownOptions = List(arg[String]("name")
    .text("Name for the TransmogrifAI project (should be in CamelCase) [required]")
    .validate(camelCase)
    .action((appName, cfg) => cfg.copy(projName = appName)),

  opt[File]("input")
    .text("Input file for the TransmogrifAI project [required]")
    .validate(fileExists)
    .required
    .action((inputFile, cfg) => cfg.copy(inputFile = Option(inputFile))),

  opt[String]("id")
    .text("Name for the ID field [required]")
    .required
    .action((id, cfg) => cfg.copy(idField = Option(id))),

  opt[String]("response")
    .text("Feature name to predict [required]")
    .required
    .action((response, cfg) => cfg.copy(response = Option(response))),

  opt[File]("answers")
    .text("answers to generator questions [optional]")
    .validate(fileExists)
    .action((answersFile, cfg) => cfg.copy(answersFile = Option(answersFile))),

  opt[Unit]("overwrite")
    .text("allowed to overwrite existing project directory")
    .action( (_, cfg) => cfg.copy(overwrite = true)),

  opt[File]("schema")
    .text("Avro schema to use for objects; required unless you specify --auto")
    .validate(fileExists)
    .action((schemaFile, cfg) =>
      cfg.copy(schemaSource = Option(AvroSchemaFromFile(schemaFile)))
    ))

  private val autoOption = opt[String]("auto")
    .text(
      "Automatic detection of data schema (you need to provide a name);\n" +
      "need this one or a schema file. Experimental feature.")
    .validate(isIdentifier)
    .action((name, cfg) => cfg.copy(schemaSource = cfg.inputFile map AutomaticSchema(name)))

  private val cliOptions = if (AUTO_ENABLED) autoOption :: knownOptions else knownOptions

  cmd("gen")
    .action((_, cfg) => cfg.copy(command = "gen"))
    .text("Generate a new TransmogrifAI project")
    .children(cliOptions.toArray: _*)

}
