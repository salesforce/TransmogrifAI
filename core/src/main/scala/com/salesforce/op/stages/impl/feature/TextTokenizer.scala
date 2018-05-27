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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.text._
import org.apache.spark.ml.param._

import scala.reflect.runtime.universe.TypeTag

trait TextTokenizerParams extends Params {

  /**
   * Indicates whether to attempt language detection.
   */
  final val autoDetectLanguage = new BooleanParam(this, "autoDetectLanguage", "whether to attempt language detection")
  def setAutoDetectLanguage(value: Boolean): this.type = set(autoDetectLanguage, value)
  def getAutoDetectLanguage: Boolean = $(autoDetectLanguage)

  /**
   * Language detection threshold.
   * If none of the detected languages have confidence greater than the threshold then defaultLanguage is used.
   */
  final val autoDetectThreshold =
    new DoubleParam(this, "autoDetectThreshold", "language detection threshold",
      ParamValidators.inRange(0.0, 1.0, true, true))
  def setAutoDetectThreshold(value: Double): this.type = set(autoDetectThreshold, value)
  def getAutoDetectThreshold: Double = $(autoDetectThreshold)

  /**
   * Default language to assume in case autoDetectLanguage is disabled or failed to make a good enough prediction.
   */
  final val defaultLanguage = new Param[String](this, "defaultLanguage", "default language")
  def setDefaultLanguage(value: Language): this.type = set(defaultLanguage, value.entryName)
  def getDefaultLanguage: Language = Language.withName($(defaultLanguage))

  /**
   * Minimum token length, >= 1.
   */
  final val minTokenLength =
    new IntParam(this, "minTokenLength", "minimum token length (>= 1)", ParamValidators.gtEq(1))
  def setMinTokenLength(value: Int): this.type = set(minTokenLength, value)
  def getMinTokenLength: Int = $(minTokenLength)

  /**
   * Indicates whether to convert all characters to lowercase before tokenizing.
   */
  final val toLowercase =
    new BooleanParam(this, "toLowercase", "whether to convert all characters to lowercase before tokenizing")
  def setToLowercase(value: Boolean): this.type = set(toLowercase, value)
  def getToLowercase: Boolean = $(toLowercase)

  setDefault(
    autoDetectLanguage -> TextTokenizer.AutoDetectLanguage,
    autoDetectThreshold -> TextTokenizer.AutoDetectThreshold,
    defaultLanguage -> TextTokenizer.DefaultLanguage.entryName,
    minTokenLength -> TextTokenizer.MinTokenLength,
    toLowercase -> TextTokenizer.ToLowercase
  )

  def tokenize(
    text: Text,
    languageDetector: LanguageDetector = TextTokenizer.LanguageDetector,
    analyzer: TextAnalyzer = TextTokenizer.Analyzer
  ): (Language, TextList) = TextTokenizer.tokenize(
    text = text,
    languageDetector = languageDetector,
    analyzer = analyzer,
    autoDetectLanguage = getAutoDetectLanguage,
    defaultLanguage = getDefaultLanguage,
    autoDetectThreshold = getAutoDetectThreshold,
    toLowercase = getToLowercase,
    minTokenLength = getMinTokenLength
  )

}

/**
 * Transformer that takes anything of type Text or lower and returns a TextList of tokens extracted from that text
 *
 * @param languageDetector a language detector instance (defaults to [[OptimaizeLanguageDetector]]
 * @param analyzer         a text analyzer instance (defaults to a [[LuceneTextAnalyzer]])
 * @param uid              uid of the stage
 */
class TextTokenizer[T <: Text]
(
  val languageDetector: LanguageDetector = TextTokenizer.LanguageDetector,
  val analyzer: TextAnalyzer = TextTokenizer.Analyzer,
  uid: String = UID[TextTokenizer[_]]
)(implicit tti: TypeTag[T])
  extends UnaryTransformer[T, TextList](operationName = "textToken", uid = uid) with TextTokenizerParams {
  def transformFn: T => TextList = text => tokenize(text, languageDetector, analyzer)._2
}

object TextTokenizer {
  val LanguageDetector: LanguageDetector = new OptimaizeLanguageDetector()
  val Analyzer: TextAnalyzer = new LuceneTextAnalyzer()
  val AnalyzerHtmlStrip: TextAnalyzer = new LuceneTextAnalyzer(LuceneTextAnalyzer.withHtmlStripping)
  val AutoDetectLanguage = false
  val AutoDetectThreshold = 0.99
  val DefaultLanguage: Language = Language.Unknown
  val MinTokenLength = 1
  val ToLowercase = true
  val StripHtml = false

  /**
   * Language wise text tokenization
   *
   * @param text                text to tokenize
   * @param languageDetector    language detector instance
   * @param analyzer            text analyzer instance
   * @param autoDetectLanguage  whether to attempt language detection
   * @param defaultLanguage     default language
   * @param autoDetectThreshold language detection threshold
   * @param toLowercase         whether to convert all characters to lowercase before tokenizing
   * @param minTokenLength      minimum token length
   * @return detected language and tokens
   */
  def tokenize(
    text: Text,
    languageDetector: LanguageDetector = LanguageDetector,
    analyzer: TextAnalyzer = Analyzer,
    autoDetectLanguage: Boolean = AutoDetectLanguage,
    defaultLanguage: Language = DefaultLanguage,
    autoDetectThreshold: Double = AutoDetectThreshold,
    toLowercase: Boolean = ToLowercase,
    minTokenLength: Int = MinTokenLength
  ): (Language, TextList) = text match {
    case SomeValue(Some(txt)) =>
      val language =
        if (!autoDetectLanguage) defaultLanguage
        else {
          languageDetector
            .detectLanguages(txt)
            .collectFirst { case (lang, confidence) if confidence > autoDetectThreshold => lang }
            .getOrElse(defaultLanguage)
        }
      val lowerTxt = if (toLowercase) txt.toLowerCase else txt
      val tokens = analyzer.analyze(lowerTxt, language)
      language -> tokens.filter(_.length >= minTokenLength).toTextList
    case _ =>
      defaultLanguage -> TextList.empty
  }
}
