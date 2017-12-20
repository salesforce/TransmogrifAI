/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import com.salesforce.op.cli.gen.FileSource.Str

import scala.util.matching.Regex
import FileGenerator._

/**
 * Encapsulates the rendering of the directives in a file (see `cli/README.md` for more on how this works).
 *
 * The [[render]] method produces a [[FileInProject]] with the final source of the file to be created in the template.
 *
 * @param path The path to render the template to.
 * @param sourceFn A delayed value so we can load the actual source for the template as late as possible
 */
class FileGenerator(val path: String, sourceFn: => String) {

  private lazy val source = sourceFn

  /**
   * Render this template with some arguments.
   *
   * @param substitutions The arguments to the template, as [[Substitutions]] (a Map[String, String])
   * @return A [[FileInProject]] that represents the rendered file to be created
   */
  def render(substitutions: Substitutions): FileInProject = {
    var directives = directiveRegex.findAllMatchIn(source).toList
    var renderedTemplate = source
    var filePath = path

    while (directives.nonEmpty) {
      val m = directives.head
      m.group(1).split(" ").toList match {
        case List("setFileName", arg) =>
          filePath = updateFileName(filePath, substitutions(arg))
          renderedTemplate = removeDirective(renderedTemplate, m)
        case List("<<", arg) =>
          renderedTemplate = splice(
            renderedTemplate, getSubstitutionEnd(renderedTemplate, m.start - 1, -1) + 1, m.end, substitutions(arg)
          )
        case List(arg, ">>") =>
          renderedTemplate = splice(
            renderedTemplate, m.start, getSubstitutionEnd(renderedTemplate, m.end, +1), substitutions(arg)
          )
        case List("replace", pattern, variable) =>
          renderedTemplate = removeDirective(renderedTemplate, m).replace(pattern, substitutions(variable))
      }
      directives = directiveRegex.findAllMatchIn(renderedTemplate).toList
    }

    FileInProject(filePath, Str(renderedTemplate))
  }

}

private[gen] object FileGenerator {

  type Substitutions = Map[String, String]

  private val spaces = Set(' ', '\n', '\t')

  private val directiveRegex = "\\/\\* ([^\\/]*) \\*\\/".r

  def splice(source: String, start: Int, end: Int, value: String): String =
    source.substring(0, start) + value + source.substring(end)

  def removeDirective(source: String, mat: Regex.Match): String = {
    splice(source, mat.start, skipWhitespace(source, mat.end + 1), "")
  }

  def updateFileName(path: String, newFileName: String): String =
    splice(path, path.lastIndexOf('/') + 1, path.lastIndexOf('.'), newFileName)

  def skipWhitespace(source: String, start: Int, inc: Int = 1): Int = {
    var i = start
    while (i > 0 && i < source.length - 1 && Set(' ', '\n', '\t').contains(source.charAt(i))) {
      i += inc
    }
    i
  }

  def skipExpr(source: String, start: Int, inc: Int = 1): Int = {

    var balance = DelimBalance(0, 0, 0)

    var i = start
    while (i > 0 && i < source.length - 1 &&
      (!spaces.contains(source.charAt(i)) || !balance.balanced)) {
      balance = source.charAt(i) match {
        case '(' => balance.copy(paren = balance.paren + 1)
        case ')' => balance.copy(paren = balance.paren - 1)
        case '[' => balance.copy(square = balance.square + 1)
        case ']' => balance.copy(square = balance.square - 1)
        case '{' => balance.copy(curly = balance.curly + 1)
        case '}' => balance.copy(curly = balance.curly - 1)
        case _ => balance
      }
      i += inc
    }
    i
  }

  private case class DelimBalance(paren: Int, square: Int, curly: Int) {
    def balanced: Boolean = paren == 0 && square == 0 && curly == 0
  }

  def getSubstitutionEnd(source: String, start: Int, inc: Int): Int = {
    val endWhitespace = skipWhitespace(source, start, inc)
    val endExpr = skipExpr(source, endWhitespace, inc)
    endExpr
  }

}


