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

import java.io._
import java.nio.file.Files

import com.salesforce.op.cli.{AvroSchemaFromFile, GeneratorConfig}
import com.salesforce.op.cli.gen.FileSource.{Str, Streaming}
import org.apache.avro.Schema

import scala.annotation.tailrec
import scala.io.Source
import scala.Console

/**
 * Does various external-related tasks
 *
 * @param config The [[GeneratorConfig]] representing a command to generate a project
 */
case class Ops(config: GeneratorConfig) {

  /**
   * Generates a project.
   */
  def run(): Unit = {
    // TODO only one (hard-coded) template for now. For future, can have multiple templates
    val templateName = "simple"
    val template = ProjectGenerator.byName(templateName)(this)
    println(s"Starting $templateName project generation")

    val generalArgs = Map("APP_NAME" -> config.projName)
    val dir = config.projectDirectory

    try {
      val templateProps = template.props
      val allSubstitutions = generalArgs ++ templateProps

      println(
        s"Generating '${config.projName}' in '${dir.getAbsolutePath}/' with template '$templateName'")
      template.renderAll(allSubstitutions) foreach {
        case FileInProject(path, source, FilePermissions(perms)) =>
          val file = new File(dir, path)
          file.getParentFile.mkdirs()
          source.writeTo(file)

          Files.setPosixFilePermissions(file.toPath, perms)
          println(s"  Created '${file.getAbsolutePath}'")
      }
      println("Done.")
      if (template.welcomeMessage != "") {
        println(template.welcomeMessage)
      }
    } catch {
      case x: Exception =>
        dir.delete() // TODO(mt, vlad): make sure that all potential contents is deleted too
        throw x
    }
  }

  private lazy val answers = {
    val map: Option[Map[String, String]] = for {
      file <- config.answersFile
    } yield {
      val kvs = for {
        line <- Source.fromFile(file).getLines
        if line contains " => "
        kv = line split " => "
      } yield kv(0).trim.toLowerCase -> kv(1)

      kvs.toMap
    }

    map getOrElse Map[String, String]()
  }

  def ask[T](question: String, options: Map[T, List[String]], errMsg: String): T = {
    Ops.ask[T](question, options, answers) getOrElse {
      throw new IllegalStateException(s"$errMsg\nThe question was: $question")
    }
  }
}

trait UserIO {
  protected def readLine(msg: String): Option[String] = Option(scala.io.StdIn.readLine(msg))

  /**
   * Prompt the user to pick between in an option list between choices of type `T`. The `T` returned is the value the
   * user picked.
   *
   * {{{
   * scala> GenTask.ask("Launch the nukes?", Map(true ->  List("yes", "doit"),
   *                                             false -> List("no", "plz no")))
   * Launch the nukes? [0] yes [1] no: 0
   * res1: Boolean = true
   * scala> GenTask.ask("Launch the nukes?", Map(true ->  List("yes", "doit"),
   *                                             false -> List("no", "plz no")))
   * Launch the nukes? [0] yes [1] no: plz no
   * res2: Boolean = false
   * }}}
   *
   * @param message The question to ask the user
   * @param options A map, mapping each `T` to a list of strings that the user could use to pick this option. The first
   *                string is special, as it is displayed to the user
   * @tparam T The type of values to pick
   * @return Which value the user picked
   */
  def ask[T](
    message: String,
    options: Map[T, List[String]],
    answers: Map[String, String] = Map.empty[String, String]): Option[T] = {
    options.collect {
      case (_, Nil) => throw new IllegalArgumentException("ask needs at least one string description per option")
    }
    // add the index to the choices list
    val optionsWithIndex = options.zipWithIndex.map {
      case ((value, choices), i) => (value, i.toString :: choices)
    }
    // the descriptions are what we present to the user
    val descriptions = optionsWithIndex.collect {
      case (_, index :: firstChoice :: _) => s"[$index] $firstChoice"
    }
    val normalizedOptions = for {
      (value, choices) <- optionsWithIndex
      choice <- choices
    } yield (choice.toLowerCase, value)
    val keys = normalizedOptions.keySet
    val q = message + descriptions.mkString(" ", " ", ": ")

    val input = qna(q, keys, answers)
    val res = input flatMap normalizedOptions.get
    res
  }

  def qna(
    question: String,
    accept: String => Boolean,
    answers: Map[String, String]
  ): Option[String] = {
    val questionSimplified = question.trim.toLowerCase
    val fromFile = answers.keySet find questionSimplified.startsWith flatMap answers.get filter accept

    var maybeAnswer = fromFile orElse readLine(question) map (_.toLowerCase.trim)

    maybeAnswer match {
      case Some(answer) =>
        // keep prompting the user until we get something
        if (accept(answer)) maybeAnswer else qna(question, accept, answers)
      case None => None
    }
  }
}

object Ops extends UserIO {

  def oops(errorMsg: String): Nothing = {
    throw new IllegalArgumentException(errorMsg)
  }
}
