/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.text

/**
 * Text analyzer trait
 */
trait TextAnalyzer extends Serializable {

  /**
   * Analyze a text and produce tokens
   *
   * @param s        input text
   * @param language suggested text language
   * @return sequence of tokens
   */
  def analyze(s: String, language: Language): Seq[String]

}
