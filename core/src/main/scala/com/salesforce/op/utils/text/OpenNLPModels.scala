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

import java.io.InputStream

import com.salesforce.op.utils.text.Language._
import com.salesforce.op.utils.text.NameEntityType._
import opennlp.tools.namefind.TokenNameFinderModel
import opennlp.tools.sentdetect.SentenceModel
import opennlp.tools.tokenize.TokenizerModel

/**
 * A factory to get/create OpenNLP models
 */
object OpenNLPModels {
  // Assumes that models are stored as a resource
  private val modelsPath = "/OpenNLP"

  private lazy val tokenNameModels: Map[(Language, NameEntityType), TokenNameFinderModel] = Map(
    (English, Date) -> loadTokenNameFinderModel(s"$modelsPath/en-ner-date.bin"),
    (English, Location) -> loadTokenNameFinderModel(s"$modelsPath/en-ner-location.bin"),
    (English, Money) -> loadTokenNameFinderModel(s"$modelsPath/en-ner-money.bin"),
    (English, Organization) -> loadTokenNameFinderModel(s"$modelsPath/en-ner-organization.bin"),
    (English, Percentage) -> loadTokenNameFinderModel(s"$modelsPath/en-ner-percentage.bin"),
    (English, Person) -> loadTokenNameFinderModel(s"$modelsPath/en-ner-person.bin"),
    (English, Time) -> loadTokenNameFinderModel(s"$modelsPath/en-ner-time.bin"),

    (Spanish, Location) -> loadTokenNameFinderModel(s"$modelsPath/es-ner-location.bin"),
    (Spanish, Organization) -> loadTokenNameFinderModel(s"$modelsPath/es-ner-organization.bin"),
    (Spanish, Person) -> loadTokenNameFinderModel(s"$modelsPath/es-ner-person.bin"),
    (Spanish, Misc) -> loadTokenNameFinderModel(s"$modelsPath/es-ner-misc.bin"),

    (Dutch, Location) -> loadTokenNameFinderModel(s"$modelsPath/nl-ner-location.bin"),
    (Dutch, Organization) -> loadTokenNameFinderModel(s"$modelsPath/nl-ner-organization.bin"),
    (Dutch, Person) -> loadTokenNameFinderModel(s"$modelsPath/nl-ner-person.bin"),
    (Dutch, Misc) -> loadTokenNameFinderModel(s"$modelsPath/nl-ner-misc.bin")
  )

  private lazy val sentenceModels: Map[Language, SentenceModel] = Map(
    Danish -> loadSentenceModel(s"$modelsPath/da-sent.bin"),
    English -> loadSentenceModel(s"$modelsPath/en-sent.bin"),
    German -> loadSentenceModel(s"$modelsPath/de-sent.bin"),
    Dutch -> loadSentenceModel(s"$modelsPath/nl-sent.bin"),
    Portuguese -> loadSentenceModel(s"$modelsPath/pt-sent.bin"),
    Sami -> loadSentenceModel(s"$modelsPath/se-sent.bin")
  )

  private lazy val tokenizerModels: Map[Language, TokenizerModel] = Map(
    Danish -> loadTokenizerModel(s"$modelsPath/da-token.bin"),
    German -> loadTokenizerModel(s"$modelsPath/de-token.bin"),
    English -> loadTokenizerModel(s"$modelsPath/en-token.bin"),
    Dutch -> loadTokenizerModel(s"$modelsPath/nl-token.bin"),
    Portuguese -> loadTokenizerModel(s"$modelsPath/pt-token.bin"),
    Sami -> loadTokenizerModel(s"$modelsPath/se-token.bin")
  )

  /**
   * Factory to get [[TokenNameFinderModel]] for a given language & entity type if it exists
   *
   * @return some [[TokenNameFinderModel]] instance or None
   */
  def getTokenNameFinderModel(language: Language, entity: NameEntityType): Option[TokenNameFinderModel] =
    tokenNameModels.get(language -> entity)

  /**
   * Factory to get [[SentenceModel]] for a given language
   *
   * @return some [[SentenceModel]] instance or None
   */
  def getSentenceModel(language: Language): Option[SentenceModel] =
    sentenceModels.get(language)

  /**
   * Factory to get [[TokenizerModel]] for a given language
   *
   * @return some [[TokenizerModel]] instance or None
   */
  def getTokenizerModel(language: Language): Option[TokenizerModel] =
    tokenizerModels.get(language)

  private def loadTokenNameFinderModel(resourcePath: String): TokenNameFinderModel = {
    val modelStream = loadFromResource(resourcePath)
    new TokenNameFinderModel(modelStream)
  }

  private def loadSentenceModel(resourcePath: String): SentenceModel = {
    val modelStream = loadFromResource(resourcePath)
    new SentenceModel(modelStream)
  }

  private def loadTokenizerModel(resourcePath: String): TokenizerModel = {
    val modelStream = loadFromResource(resourcePath)
    new TokenizerModel(modelStream)
  }

  private def loadFromResource(resourcePath: String): InputStream =
    try {
      getClass.getResourceAsStream(resourcePath)
    } catch {
      case e: Exception => throw new RuntimeException(
        s"Failed to load OpenNLP model from resource '$resourcePath'. " +
          "Make sure to include OP 'models' dependency jar in your application classpath.", e
      )
    }

}
