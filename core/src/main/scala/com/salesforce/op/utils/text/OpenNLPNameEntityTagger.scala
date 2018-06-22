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

import com.salesforce.op.utils.text.NameEntityType._
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import opennlp.tools.namefind.NameFinderME
import opennlp.tools.util.Span

/**
 * OpenNLP implementation of [[NameEntityTagger]]
 */
class OpenNLPNameEntityTagger extends NameEntityTagger[OpenNLPTagResult] {

  /**
   * Apply the name entity recognition model on the sentence tokens to retrieve information
   *
   * @param tokens        sentence tokens
   * @param language      language
   * @param entitiesToTag entities to tag if found
   * @return map of entity and corresponding tokens
   */
  def tag(
    tokens: Seq[String],
    language: Language,
    entitiesToTag: Seq[NameEntityType]
  ): OpenNLPTagResult = {
    val tokensArr = tokens.toArray
    val empty = Map.empty[String, Set[NameEntityType]]
    val tags = entitiesToTag.foldLeft(empty) { (acc, entityToTag) =>
      OpenNLPModels.getTokenNameFinderModel(language, entityToTag) match {
        case None => acc
        case Some(model) =>
          val finder = new NameFinderME(model)
          val spans = finder.find(tokensArr)
          val res = convertSpansToMap(spans, tokensArr)
          acc + res
      }
    }
    OpenNLPTagResult(tags)
  }

  /**
   * Retrieve information from the model output
   *
   * @param spans  open nlp name entity finder model output
   * @param tokens sentence tokens
   * @return map of token and its tag set
   */
  private[op] def convertSpansToMap(spans: Seq[Span], tokens: Array[String]): Map[String, Set[NameEntityType]] = {
    // span objects provide exclusive end index
    val pairSeq = for {
      span <- spans
      entity = Seq(nameEntityType(span.getType.toLowerCase))
      token <- tokens.slice(span.getStart, span.getEnd)
    } yield token -> entity

    // aggregate results by token convert the output to map
    pairSeq
      .groupBy { case (token, _) => token }
      .map { case (token, entities) =>
        token -> entities.flatMap(_._2).toSet
      }
  }

  private def nameEntityType: String => NameEntityType = {
    case "date" => Date
    case "location" => Location
    case "money" => Money
    case "organization" => Organization
    case "percentage" => Percentage
    case "person" => Person
    case "time" => Time
    case "misc" => Misc
    case _ => Other
  }
}


/**
 * OpenNLP implementation of [[TaggerResult]]
 *
 * @param tokenTags token tags map, where keys are token and values are entities matching each token
 */
case class OpenNLPTagResult(tokenTags: Map[String, Set[NameEntityType]]) extends TaggerResult
