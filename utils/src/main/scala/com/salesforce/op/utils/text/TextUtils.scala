/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.text


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

}
