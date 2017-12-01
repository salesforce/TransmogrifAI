/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import java.io.InputStream

import com.salesforce.op.cli.GeneratorConfig
import com.salesforce.op.cli.gen.FileSource.{Str, Streaming}
import com.salesforce.op.cli.gen.templates.SimpleProject
import org.reflections.Reflections
import org.reflections.scanners.ResourcesScanner
import org.scalafmt.Scalafmt
import org.scalafmt.config.ScalafmtConfig

import collection.JavaConverters._
import scala.io.Source

trait ProjectTemplate {

  /**
   * The command that generates this project template.
   */
  def conf: GeneratorConfig

  /**
   * Name of the project template, exposed to user.
   */
  def name: String

  /**
   * The welcome message to be printed out to the user.
   */
  def welcomeMessage: String

  /**
   * Perform any last replacements on the file.
   */
  def replace(file: ProjectFile): ProjectFile

  /**
   * Generates arguments to feed into the templates.
   *
   * @return Arguments as [[FileTemplate.Substitutions]]
   */
  def props: FileTemplate.Substitutions

  /**
   * Returns if the file should be copied instead of rendered
   */
  def shouldCopy(path: String): Boolean = false

  private lazy val templateResourcePath = s"templates/$name/"

  lazy val templateResources: List[String] =
    new Reflections(s"templates.$name", new ResourcesScanner)
      .getResources(".*".r.pattern)
      .asScala.toList.map(_.replace(templateResourcePath, ""))

  private def loadResource(templatePath: String): InputStream = {
    val resourcePath = s"/$templateResourcePath$templatePath"
    getClass.getResourceAsStream(resourcePath)
  }

  private def loadTemplateFileSource(templatePath: String): String =
    Source.fromInputStream(loadResource(templatePath)).getLines.map(_ + "\n").mkString

  private lazy val templates: Map[String, FileTemplate] = templateResources.filterNot(shouldCopy).map { path =>
    path -> new FileTemplate(path, loadTemplateFileSource(path))
  }.toMap

  private lazy val copies: List[ProjectFile] = templateResources.filter(shouldCopy).map { path =>
    ProjectFile(path = path, source = Streaming(loadResource(path)))
  }

  def templateFile(name: String): FileTemplate = templates(name)

  def renderAll(
    substitutions: FileTemplate.Substitutions, formatScala: Boolean = true
  ): Traversable[ProjectFile] = {

    val rendered = templates map {
      case (_, tpl) => tpl.render(substitutions)
    }

    val formatted =
      if (formatScala) rendered.map {
        case f @ ProjectFile(path, Str(source), _) if path.endsWith(".scala") =>
          f.copy(source = Str(Scalafmt.format(source, new ScalafmtConfig(maxColumn = 120)).get))
        case f => f
      }
      else rendered

    val allFiles = formatted ++ copies

    allFiles map replace
  }

}

object ProjectTemplate {
  val byName: Map[String, Ops => ProjectTemplate] =
    Map(
    "simple" -> SimpleProject
  )

}
