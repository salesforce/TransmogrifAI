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

import enumeratum.{Enum, EnumEntry}

/**
 * Interface for Name Entity Recognition tagger
 *
 * @tparam Result result of the [[NameEntityTagger.tag]] function call
 */
trait NameEntityTagger[Result <: TaggerResult] extends Serializable {

  /**
   * Apply the name entity recognition model on the sentence tokens to retrieve information
   *
   * @param tokens        sentence tokens
   * @param language      language
   * @param entitiesToTag entities to tag if found
   * @return map of entity and corresponding tokens
   */
  def tag(tokens: Seq[String], language: Language, entitiesToTag: Seq[NameEntityType]): Result

}

/**
 * Result of [[NameEntityTagger.tag]] function call
 */
trait TaggerResult extends Serializable {

  /**
   * Result must be convertible to Map,
   * where keys are token and values are entities matching each token
   */
  def tokenTags: Map[String, Set[NameEntityType]]

}


/**
 * Name Entity Recognition entity type
 */
sealed trait NameEntityType extends EnumEntry with Serializable

/**
 * Name Entity Recognition entity type
 */
object NameEntityType extends Enum[NameEntityType] {
  val values = findValues
  case object Date extends NameEntityType
  case object Location extends NameEntityType
  case object Money extends NameEntityType
  case object Organization extends NameEntityType
  case object Percentage extends NameEntityType
  case object Person extends NameEntityType
  case object Time extends NameEntityType
  case object Misc extends NameEntityType
  case object Other extends NameEntityType
}
