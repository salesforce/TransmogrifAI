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

package com.salesforce.op.utils.text

import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization


case object TextUtils {
  val defaultCleanParams = CleanTextParams(ignoreCase = true, cleanPunctuations = true)

  def cleanOptString(raw: Option[String], splitOn: String = " ",
    cleanTextParams: CleanTextParams = defaultCleanParams): Option[String] =
    raw.map(t => cleanString(t, splitOn, cleanTextParams))

  def cleanString(raw: String, splitOn: String = " ", cleanTextParams: CleanTextParams = defaultCleanParams): String = {
    val l = if (cleanTextParams.ignoreCase) raw.toLowerCase else raw
    if (cleanTextParams.cleanPunctuations) {
      l
      .replaceAll("[\\p{Punct}]", splitOn)
        .replaceAll(s"$splitOn+", s"$splitOn")
        .split(splitOn)
        .map(w => w.capitalize)
        .mkString("")
    }
    else {
      l
    }
  }

  def concat(l: String, r: String, separator: String): String =
    if (l.isEmpty) r else if (r.isEmpty) l else s"$l$separator$r"

}

case class CleanTextParams(ignoreCase: Boolean, cleanPunctuations: Boolean)

object CleanTextParams {
  def jsonEncode(value: CleanTextParams): String = {
    implicit val formats = DefaultFormats
    Serialization.write(value)
  }

  def jsonDecode(json: String): CleanTextParams = {
    implicit val formats = DefaultFormats
    Serialization.read[CleanTextParams](json)
  }
}
