/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
