/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.text._
import org.apache.spark.ml.param._

import scala.reflect.runtime.universe.TypeTag

/**
 * Transformer that takes anything of type Text or lower and returns a TextList of tokens extracted from that text
 *
 * @param languageDetector a language detector instance (defaults to [[OptimaizeLanguageDetector]]
 * @param analyzer a text analyzer instance (defaults to a [[LuceneTextAnalyzer]])
 * @param uid      uid of the stage
 */
class TextTokenizer[T <: Text]
(
  val languageDetector: LanguageDetector = TextTokenizer.LanguageDetector,
  val analyzer: TextAnalyzer = TextTokenizer.Analyzer,
  uid: String = UID[TextTokenizer[_]]
)(implicit tti: TypeTag[T])
  extends UnaryTransformer[T, TextList](operationName = "textToken", uid = uid) {

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

  /**
   * Function used to convert input to output
   */
  override def transformFn: T => TextList = text => {
    if (text.isEmpty) TextList.empty
    else {
      val language = {
        if (!$(autoDetectLanguage)) getDefaultLanguage
        else {
          val threshold = $(autoDetectThreshold)
          languageDetector
            .detectLanguages(text.v.get)
            .collect { case (lang, confidence) if confidence > threshold => lang }
            .headOption.getOrElse(getDefaultLanguage)
        }
      }
      val txt = if ($(toLowercase)) text.v.get.toLowerCase else text.v.get
      val tokens = analyzer.analyze(txt, language)
      val minTokLen = $(minTokenLength)
      tokens.filter(_.length >= minTokLen).toTextList
    }
  }

}

object TextTokenizer {
  val LanguageDetector: LanguageDetector = new OptimaizeLanguageDetector()
  val Analyzer: TextAnalyzer = new LuceneTextAnalyzer()
  val AutoDetectLanguage = false
  val AutoDetectThreshold = 0.99
  val DefaultLanguage = Language.Unknown
  val MinTokenLength = 1
  val ToLowercase = true
}
