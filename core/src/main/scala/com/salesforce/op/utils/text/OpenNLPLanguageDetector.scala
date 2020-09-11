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
import opennlp.tools.ml.model.MaxentModel
import opennlp.tools.langdetect.{LanguageDetectorContextGenerator, LanguageDetectorME, LanguageDetectorModel}
import org.slf4j.LoggerFactory

class OpenNLPLanguageDetector extends LanguageDetector {
  /**
   * Detect languages from a text
   *
   * @param s input text
   * @return detected languages sorted by confidence score in descending order.
   *         Confidence score is range of [0.0, 1.0], with higher values implying greater confidence.
   */
  def detectLanguages(s: String): Seq[(Language, Double)] = {
    OpenNLPLanguageDetector.detector.predict(s)
  }
}


case class OpenNLPLanguageDetectorME(
  languageDetectorModel: MaxentModel,
  contextGenerator: LanguageDetectorContextGenerator
) {
  def predict(str: String): Seq[(Language, Double)] = {
    languageDetectorModel
      .eval(contextGenerator.getContext(str))
      .zipWithIndex
      .sortBy { case (confidence, _) => confidence }
      .reverse
      .map { case (prob, index) =>
        (Language.fromString(languageDetectorModel.getOutcome(index)), prob)
      }
  }
}


private[op] object OpenNLPLanguageDetector {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  // This detector is a singleton to avoid reloading the ngrams for the detector
  lazy val detector = {
    val start = System.currentTimeMillis()
    val ldm = OpenNLPModels.getLanguageDetection()
    val model = ldm.getMaxentModel
    val contextGenerator = ldm.getFactory.getContextGenerator
    println(s"GERA DEBUG Loaded OpenNLP Language Model for ${model.getNumOutcomes} languages. " +
      s"Time elapsed: ${System.currentTimeMillis() - start}ms")
    OpenNLPLanguageDetectorME(model, contextGenerator)
  }
}


