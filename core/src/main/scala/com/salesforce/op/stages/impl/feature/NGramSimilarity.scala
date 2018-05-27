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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryTransformer
import org.apache.lucene.search.spell.NGramDistance

import scala.reflect.runtime.universe.TypeTag


/**
 * Compute char ngram distance for MultiPickList features.
 *
 * @param nGramSize the size of the n-gram to be used to compute the string distance
 */
class SetNGramSimilarity
(
  nGramSize: Int = NGramSimilarity.nGramSize,
  uid: String = UID[SetNGramSimilarity]
) extends NGramSimilarity[MultiPickList](
  operationName = "nGramSet",
  uid = uid,
  convertFn = _.v.mkString(" "),
  nGramSize = nGramSize
)

/**
 * Compute char ngram distance for Text features.
 *
 * @param nGramSize the size of the n-gram to be used to compute the string distance
 */
class TextNGramSimilarity[T <: Text]
(
  nGramSize: Int = NGramSimilarity.nGramSize,
  uid: String = UID[TextNGramSimilarity[T]]
)(implicit tti1: TypeTag[T]) extends NGramSimilarity[T](
  operationName = "nGramText",
  uid = uid,
  convertFn = _.v.getOrElse(""),
  nGramSize = nGramSize
)

private[feature] class NGramSimilarity[I <: FeatureType]
(
  uid: String,
  operationName: String,
  val convertFn: I => String,
  val nGramSize: Int
)(implicit tti1: TypeTag[I],
  tto: TypeTag[Real]
) extends BinaryTransformer[I, I, RealNN](operationName = operationName, uid = uid) {

  def transformFn: (I, I) => RealNN = (lhs: I, rhs: I) => {
    val lhString = convertFn(lhs).trim
    val rhString = convertFn(rhs).trim

    // in our case, if any of the strings are empty, we want the similarity to be minimum, not maximum,
    // regardless of what the other string is.
    val similarity = {
      if (lhString.isEmpty || rhString.isEmpty) 0.0
      else new NGramDistance(nGramSize).getDistance(lhString, rhString)
    }
    similarity.toRealNN
  }

}

object NGramSimilarity {
  val nGramSize = 3
}
