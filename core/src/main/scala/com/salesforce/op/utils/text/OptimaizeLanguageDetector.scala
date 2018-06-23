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

import com.optimaize.langdetect.LanguageDetectorBuilder
import com.optimaize.langdetect.i18n.LdLocale
import com.optimaize.langdetect.ngram.NgramExtractors
import com.optimaize.langdetect.profiles.LanguageProfileReader
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
