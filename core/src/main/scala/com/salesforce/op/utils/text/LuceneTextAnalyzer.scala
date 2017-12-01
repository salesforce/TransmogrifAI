/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.text

import com.salesforce.op.utils.text.Language._
import org.apache.lucene.analysis.ar.ArabicAnalyzer
import org.apache.lucene.analysis.bg.BulgarianAnalyzer
import org.apache.lucene.analysis.ca.CatalanAnalyzer
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
import org.apache.lucene.analysis.{Analyzer, TokenStream}

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

  /**
   * Creates a Lucene Analyzer for a specific language or falls back to [[StandardAnalyzer]]
   *
   * @param lang desired language
   * @return language specific language analyzer or [[StandardAnalyzer]] as default
   */
  def apply(lang: Language): Analyzer = analyzers.getOrElse(lang, DefaultAnalyzer)

}
