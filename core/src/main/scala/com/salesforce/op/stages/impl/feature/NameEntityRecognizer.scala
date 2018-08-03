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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.text._
import com.twitter.algebird.Operators._

import scala.reflect.runtime.universe.TypeTag

/**
 * Name Entity [[NameEntityType]] text recognizer.
 *
 * Note: when providing your own the analyzer/splitter/tagger make sure they can work together,
 * for instance OpenNLP models require their own analyzers to be provided when tokenizing.
 * The returned feature type is a [[MultiPickListMap]] which contains sets of entities for all the tokens
 *
 * @param languageDetector a language detector instance (defaults to [[OptimaizeLanguageDetector]]
 * @param analyzer         a text analyzer instance (defaults to a [[OpenNLPAnalyzer]])
 * @param sentenceSplitter a sentence splitter instance (defaults to a [[OpenNLPSentenceSplitter]])
 * @param tagger           name entity tagger (defaults to [[OpenNLPNameEntityTagger]])
 * @param uid              uid for instance
 * @param tti              type tag for input feature type
 * @tparam T text feature type
 */
class NameEntityRecognizer[T <: Text]
(
  val languageDetector: LanguageDetector = NameEntityRecognizer.LanguageDetector,
  val analyzer: TextAnalyzer = NameEntityRecognizer.Analyzer,
  val sentenceSplitter: SentenceSplitter = NameEntityRecognizer.Splitter,
  val tagger: NameEntityTagger[_ <: TaggerResult] = NameEntityRecognizer.Tagger,
  uid: String = UID[NameEntityRecognizer[_]]
)(implicit tti: TypeTag[T])
  extends UnaryTransformer[T, MultiPickListMap](uid = uid, operationName = "nameEntityRec")
    with LanguageDetectionParams {

  setDefault(
    autoDetectLanguage -> NameEntityRecognizer.AutoDetectLanguage,
    autoDetectThreshold -> NameEntityRecognizer.AutoDetectThreshold,
    defaultLanguage -> NameEntityRecognizer.DefaultLanguage.entryName
  )

  def transformFn: T => MultiPickListMap = text => {
    val res = TextTokenizer.tokenize(
      text = text,
      languageDetector = languageDetector,
      analyzer = analyzer,
      sentenceSplitter = Option(sentenceSplitter),
      autoDetectLanguage = getAutoDetectLanguage,
      autoDetectThreshold = getAutoDetectThreshold,
      defaultLanguage = getDefaultLanguage,
      toLowercase = false
    )
    val sentenceTags = res.sentences.view.map { sentence =>
      val tags = tagger.tag(sentence.value, res.language, NameEntityType.values)
      tags.tokenTags.mapValues(_.map(_.toString))
    }
    sentenceTags.foldLeft(Map.empty[String, Set[String]])(_ + _).toMultiPickListMap
  }

}

object NameEntityRecognizer {
  val Analyzer: TextAnalyzer = new OpenNLPAnalyzer()
  val LanguageDetector: LanguageDetector = new OptimaizeLanguageDetector()
  val Tagger: NameEntityTagger[_ <: TaggerResult] = new OpenNLPNameEntityTagger()
  val Splitter: SentenceSplitter = new OpenNLPSentenceSplitter()
  val AutoDetectLanguage = false
  val AutoDetectThreshold = 0.99
  val DefaultLanguage: Language = Language.English
}
