/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import java.io._
import java.nio.file.Files

import com.salesforce.op.cli.GeneratorConfig
import com.salesforce.op.cli.gen.FileSource.{Str, Streaming}
import org.apache.avro.Schema

import scala.annotation.tailrec
import scala.io.{Source, StdIn}

/**
 * Does various external-related tasks
 *
 * @param config The [[GeneratorConfig]] representing a command to generate a project
 */
case class Ops(config: GeneratorConfig) {
  /**
   * The [[ProblemSchema]] for this command to generate a project. This is a lazy val so we don't prompt the user
   * until it is actually needed.
   */
  lazy val schema: ProblemSchema =
    ProblemSchema.from(this, config.schemaFile, config.response, config.idField)



  /**
   * Generates a project.
   */
  def run(): Unit = {

    // TODO only one (hard-coded) template for now. For future, can have multiple templates
    val templateName = "simple"
    val template = ProjectTemplate.byName(templateName)(this)

    val generalArgs = Map("APP_NAME" -> config.projName)
    val templateProps = template.props
    val allSubstitutions = generalArgs ++ templateProps

    val dir = config.projectDirectory
    println(s"Generating '${config.projName}' in '${dir.getAbsolutePath}/' with template '$templateName'")

    template.renderAll(allSubstitutions).foreach {
      case ProjectFile(path, source, FilePermissions(perms)) =>
        val file = new File(dir, path)
        file.getParentFile.mkdirs()

        source match {
          case Str(s) =>
            val writer = new FileWriter(file)
            writer.write(s)
            writer.close()
          case Streaming(inputStream) => Files.copy(inputStream, file.toPath)
        }

        Files.setPosixFilePermissions(file.toPath, perms)
        println(s"Created '${file.getAbsolutePath}'")
    }
    println("Done.")
    if (template.welcomeMessage != "") {
      println(template.welcomeMessage)
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
      } yield kv(0).trim -> kv(1)

      kvs.toMap
    }

    map getOrElse Map[String, String]()
  }

  def ask[T](message: String, options: Map[T, List[String]]): T = {
    Ops.ask[T](message, options, answers)
  }
}

trait Console {
  protected def readLine(msg: String): String = StdIn.readLine(msg)

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
    answers: Map[String, String] = Map.empty[String, String]): T = {
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
    // keep prompting the user until we get something
    val input = qna(q, keys, answers)
    normalizedOptions(input)
  }

  def qna(
    question: String,
    accept: String => Boolean,
    answers: Map[String, String]
  ): String = {
    var answer = answers.get(question.trim) filter accept getOrElse readLine(question).toLowerCase

    if (accept(answer)) answer else qna(question, accept, answers)
  }
}

object Ops extends Console {

  def oops(errorMsg: String): Nothing = {
    throw new IllegalArgumentException(errorMsg)
  }


}
