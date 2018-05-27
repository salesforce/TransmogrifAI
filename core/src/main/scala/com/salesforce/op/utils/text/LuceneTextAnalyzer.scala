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

package com.salesforce.op.utils.text

import java.io.Reader

import com.salesforce.op.utils.text.Language._
import org.apache.lucene.analysis.ar.ArabicAnalyzer
import org.apache.lucene.analysis.bg.BulgarianAnalyzer
import org.apache.lucene.analysis.ca.CatalanAnalyzer
import org.apache.lucene.analysis.charfilter.HTMLStripCharFilter
import org.apache.lucene.analysis.cjk.CJKAnalyzer
import org.apache.lucene.analysis.cz.CzechAnalyzer
import org.apache.lucene.analysis.da.DanishAnalyzer
import org.apache.lucene.analysis.de.GermanAnalyzer
import org.apache.lucene.analysis.el.GreekAnalyzer
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.analysis.es.SpanishAnalyzer
import org.apache.lucene.analysis.eu.BasqueAnalyzer
import org.apache.lucene.analysis.fa.PersianAnalyzer
import org.apache.lucene.analysis.fi.FinnishAnalyzer
import org.apache.lucene.analysis.fr.FrenchAnalyzer
import org.apache.lucene.analysis.ga.IrishAnalyzer
import org.apache.lucene.analysis.gl.GalicianAnalyzer
import org.apache.lucene.analysis.hi.HindiAnalyzer
import org.apache.lucene.analysis.hu.HungarianAnalyzer
import org.apache.lucene.analysis.id.IndonesianAnalyzer
import org.apache.lucene.analysis.it.ItalianAnalyzer
import org.apache.lucene.analysis.ja.JapaneseAnalyzer
import org.apache.lucene.analysis.lt.LithuanianAnalyzer
import org.apache.lucene.analysis.lv.LatvianAnalyzer
import org.apache.lucene.analysis.nl.DutchAnalyzer
import org.apache.lucene.analysis.no.NorwegianAnalyzer
import org.apache.lucene.analysis.pt.PortugueseAnalyzer
import org.apache.lucene.analysis.ro.RomanianAnalyzer
import org.apache.lucene.analysis.ru.RussianAnalyzer
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.analysis.sv.SwedishAnalyzer
import org.apache.lucene.analysis.th.ThaiAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.lucene.analysis.tr.TurkishAnalyzer
import org.apache.lucene.analysis.{Analyzer, AnalyzerWrapper, TokenStream}

import scala.collection.mutable.ArrayBuffer

/**
 * Text analyzer implementation using a Lucene analyzer
 *
 * @param analyzers Lucene analyzer factory to use (defaults to [[LuceneTextAnalyzer]])
 */
class LuceneTextAnalyzer
(
  // use lambda to workaround a non serializable analyzer
  analyzers: Language => Analyzer = LuceneTextAnalyzer.apply
) extends TextAnalyzer {

  /**
   * Analyze a text and produce tokens
   *
   * @param s        input text
   * @param language assumed text language
   * @return sequence of tokens
   */
  def analyze(s: String, language: Language): Seq[String] = {
    val tokens = ArrayBuffer.empty[String]
    var tokenStream: TokenStream = null
    try {
      val analyzer = analyzers(language)
      tokenStream = analyzer.tokenStream(null, s)
      val token = tokenStream.addAttribute(classOf[CharTermAttribute])
      tokenStream.reset()
      while (tokenStream.incrementToken()) tokens += new String(token.buffer(), 0, token.length())
      tokenStream.end()
    } finally {
      if (tokenStream != null) {
        // close token stream safely
        try tokenStream.close() catch {
          case _: Exception =>
        }
      }
    }
    tokens
  }

}

/**
 * Creates a Lucene Analyzer for a specific language or falls back to [[StandardAnalyzer]]
 */
object LuceneTextAnalyzer {

  /**
   * Default analyzer to use if a language specific one is not present
   */
  val DefaultAnalyzer: Analyzer = new StandardAnalyzer()

  // TODO we should add specific analyzers per each language if possible
  private val analyzers: Map[Language, Analyzer] = Map(
    Arabic -> new ArabicAnalyzer(),
    Catalan -> new CatalanAnalyzer(),
    Bulgarian -> new BulgarianAnalyzer(),
    Czech -> new CzechAnalyzer(),
    Danish -> new DanishAnalyzer(),
    German -> new GermanAnalyzer(),
    Greek -> new GreekAnalyzer(),
    English -> new EnglishAnalyzer(),
    Spanish -> new SpanishAnalyzer(),
    Basque -> new BasqueAnalyzer(),
    Persian -> new PersianAnalyzer(),
    Finnish -> new FinnishAnalyzer(),
    French -> new FrenchAnalyzer(),
    Irish -> new IrishAnalyzer(),
    Galician -> new GalicianAnalyzer(),
    Hindi -> new HindiAnalyzer(),
    Hungarian -> new HungarianAnalyzer(),
    Indonesian -> new IndonesianAnalyzer(),
    Italian -> new ItalianAnalyzer(),
    Japanese -> new JapaneseAnalyzer(),
    Korean -> new CJKAnalyzer(),
    Lithuanian -> new LithuanianAnalyzer(),
    Latvian -> new LatvianAnalyzer(),
    Dutch -> new DutchAnalyzer(),
    Norwegian -> new NorwegianAnalyzer(),
    Portuguese -> new PortugueseAnalyzer(),
    Romanian -> new RomanianAnalyzer(),
    Russian -> new RussianAnalyzer(),
    Swedish -> new SwedishAnalyzer(),
    Thai -> new ThaiAnalyzer(),
    Turkish -> new TurkishAnalyzer(),
    SimplifiedChinese -> new CJKAnalyzer(),
    TraditionalChinese -> new CJKAnalyzer()
  )

  private val defaultAnalyzerHtmlStrip = stripHtml(DefaultAnalyzer)

  private val analyzersHtmlStrip = analyzers.map { case (lang, analyzer) => lang -> stripHtml(analyzer) }

  private def stripHtml(analyzer: Analyzer): Analyzer =
    new AnalyzerWrapper(analyzer.getReuseStrategy) {
      override def getWrappedAnalyzer(fieldName: String): Analyzer = analyzer
      override def wrapReader(fieldName: String, reader: Reader) = new HTMLStripCharFilter(reader)
    }

  /**
   * Creates a Lucene Analyzer for a specific language or falls back to [[StandardAnalyzer]]
   *
   * @param lang desired language
   * @return language specific language analyzer or [[StandardAnalyzer]] as default
   */
  def apply(lang: Language): Analyzer = analyzers.getOrElse(lang, DefaultAnalyzer)

  /**
   * Creates a Lucene Analyzer for a specific language or falls back to [[StandardAnalyzer]]
   * with HTML stripping applied [[HTMLStripCharFilter]]
   *
   * @param lang desired language
   * @return language specific language analyzer or [[StandardAnalyzer]] as default
   *         with HTML stripping applied [[HTMLStripCharFilter]]
   */
  def withHtmlStripping(lang: Language): Analyzer = analyzersHtmlStrip.getOrElse(lang, defaultAnalyzerHtmlStrip)

}
