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

trait ProjectGenerator {

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
  def replace(file: FileInProject): FileInProject

  /**
   * Generates arguments to feed into the templates.
   *
   * @return Arguments as [[FileGenerator.Substitutions]]
   */
  def props: FileGenerator.Substitutions

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

  private lazy val templates: Map[String, FileGenerator] = templateResources.filterNot(shouldCopy).map { path =>
    path -> new FileGenerator(path, loadTemplateFileSource(path))
  }.toMap

  private lazy val copies: List[FileInProject] = templateResources.filter(shouldCopy).map { path =>
    FileInProject(path = path, source = Streaming(loadResource(path)))
  }

  def templateFile(name: String): FileGenerator = templates(name)

  def renderAll(
    substitutions: FileGenerator.Substitutions, formatScala: Boolean = true
  ): Traversable[FileInProject] = {

    val rendered = templates map {
      case (_, tpl) => tpl.render(substitutions)
    }

    val formatted =
      if (formatScala) rendered map {
        case f @ FileInProject(path, Str(source), _) if path.endsWith(".scala") =>
          f.copy(source = Str(Scalafmt.format(source, new ScalafmtConfig(maxColumn = 120)).get))
        case f => f
      }
      else rendered

    val allFiles = formatted ++ copies

    allFiles map replace
  }

}

object ProjectGenerator {
  val byName: Map[String, Ops => ProjectGenerator] =
    Map(
    "simple" -> SimpleProject
  )

}
