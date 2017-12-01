/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.text

import com.optimaize.langdetect.i18n.LdLocale
import com.optimaize.langdetect.ngram.NgramExtractors
import com.optimaize.langdetect.profiles.LanguageProfileReader
import com.optimaize.langdetect.{LanguageDetectorBuilder, LanguageDetector => OLanguageDetector}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

/**
 * Language detector implementation using Optimaize/language-detector library
 * [[https://github.com/optimaize/language-detector]]
 */
class OptimaizeLanguageDetector extends LanguageDetector {
  /**
   * Detect languages from a text
   *
   * @param s input text
   * @return detected languages sorted by confidence score in descending order.
   *         Confidence score is range of [0.0, 1.0], with higher values implying greater confidence.
   */
  def detectLanguages(s: String): Seq[(Language, Double)] = {
    OptimaizeLanguageDetector.detector.getProbabilities(s).asScala
      .map(r => makeLanguage(r.getLocale) -> r.getProbability)
      .sortBy(-_._2)
  }

  private def makeLanguage(locale: LdLocale): Language = {
    val maybeRegion = if (locale.getRegion.isPresent) s"-${locale.getRegion.get()}" else ""
    Language.withNameInsensitive(s"${locale.getLanguage}$maybeRegion")
  }

}

private[op] object OptimaizeLanguageDetector {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  // This detector is a singleton to avoid reloading the ngrams for the detector
  lazy val detector = {
    val start = System.currentTimeMillis()
    val langs = new LanguageProfileReader().readAllBuiltIn()
    val detctr = LanguageDetectorBuilder
      .create(NgramExtractors.standard())
      .withProfiles(langs)
      .build()
    if (log.isDebugEnabled) {
      log.debug(s"Loaded {} languages. Time elapsed: {}ms", langs.size(), System.currentTimeMillis() - start)
    }
    detctr
  }
}
