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

import scala.util.matching.Regex


case object TextUtils {

  def cleanOptString(raw: Option[String], splitOn: String = " "): Option[String] =
    raw.map(t => cleanString(t, splitOn))

  def cleanString(raw: String, splitOn: String = " "): String = {
    raw
      .toLowerCase
      .replaceAll("[\\p{Punct}]", splitOn)
      .replaceAll(s"$splitOn+", s"$splitOn")
      .split(splitOn)
      .map(w => w.capitalize)
      .mkString("")
  }

  def concat(l: String, r: String, separator: String): String =
    if (l.isEmpty) r else if (r.isEmpty) l else s"$l$separator$r"

  /**
   * Helper function to conditionally extract groups from Text using RegEx.
   * @param patterns: Seq[Regex] where earlier entries are preferred to later ones;
   *                each RegEx must contain exactly one matching group
   * @param string: the string to find matches in
   * @return the first matched group found, an empty string if no matches were found
   */
  def getBestRegexMatch(patterns: Seq[Regex], string: String): String = {
    patterns.foldLeft("")({ (acc: String, pattern: Regex) =>
      if (acc == "") string match {
        case pattern(possibleZip) => possibleZip
        case _ => ""
      }
      else acc
    })
  }
}
