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
    var renderedTemplate = replaceMultiline(source, substitutions)
    var directives = directiveRegex.findAllMatchIn(source).toList
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

        case List("replaceLine", arg) =>
          renderedTemplate = splice(
            renderedTemplate,
            whereLineStarts(renderedTemplate, m.start),
            whereLineEnds(renderedTemplate, m.end) + 1,
            substitutions(arg)
          )
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

  def whereLineStarts(source: String, pos: Int): Int = {
    for {
      i <- math.min(math.max(0, pos - 1), source.length) until (0, -1)
      if source(i) == '\n'
    } return i + 1
    0
  }

  def whereLineEnds(source: String, pos: Int): Int = {
    for {
      i <- math.min(pos, source.length) until source.length
      if source(i) == '\n'
    } return i - 1
    source.length - 1
  }

  def skipWhitespace(source: String, start: Int, inc: Int = 1): Int = {
    var i = start
    while (i > 0 && i < source.length - 1 && " \n\t".contains(source(i))) {
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

  private trait Chunk {
    def append(line: String): Chunk
    def render(substitutions: Substitutions): String
  }

  private case class Lines(lines: List[String] = Nil) extends Chunk {
    def append(line: String): Lines = Lines(line::lines)
    def render(substitutions: Substitutions): String = {
      lines.reverse mkString "\n"
    }
  }

  private case class Fragment(name: String) extends Chunk {
    def append(line: String): Fragment = this
    def render(substitutions: Substitutions): String = {
      substitutions(name)
    }
  }

  def replaceMultiline(in: String, substitutions: Substitutions): String = {
    if (in contains "BEGIN") {
      val MacroStart = "\\s*//\\s+BEGIN\\s+(\\w+).*".r
      val MacroEnd = "\\s*//\\s+END\\s+(\\w+).*".r
      val lines = in split "\n"
      val chunks = (List[Chunk](Lines()) /: lines) {
        case ((ls: Lines)::t, MacroStart(name)) => Fragment(name)::ls::t
        case ((ls: Lines)::t, MacroEnd(name)) =>
          throw new IllegalArgumentException(s"Bad template $in: // END $name before //BEGIN")
        case ((ls: Lines)::t, line) => ls.append(line)::t
        case ((Fragment(name1)::t), MacroStart(name2)) =>
          throw new IllegalArgumentException(s"Unexpected template $name2 inside template $name1")
        case (cs@(Fragment(name1)::t), MacroEnd(name2)) =>
          if (name1 != name2) {
            throw new IllegalArgumentException(s"Unexpected template end $name2 inside template $name1")
          } else Lines()::cs
        case (cs@((_: Fragment)::t), line) => cs
      }
      val out = chunks.reverse map (_ render substitutions) mkString "\n"
      out
    } else {
      in
    }
  }

}


