/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.feature._

import scala.reflect.runtime.universe._


trait RichSetFeature {
  self: RichVectorFeature =>

  /**
   * Enrichment functions for OPSet Feature
   *
   * @param f OPSet Feature
   */
  implicit class RichOPSetFeature[T <: OPSet[_] : TypeTag](val f: FeatureLike[T])
    (implicit ttiv: TypeTag[T#Value]) {

    /**
     * Converts a sequence of OPSet features into a vector keeping the top K most common occurrences of each
     * OPSet feature (ie the final vector has length k * number of OPSet inputs). Plus an additional column
     * for "other" values - which will capture values that do not make the cut or values not seen in training
     *
     * @param others     other features to include in the pivot
     * @param topK       keep topK values
     * @param minSupport min occurances to keep a value
     * @param cleanText  if true ignores capitalization and punctuations when grouping categories
     * @param trackNulls keep a count of nulls
     * @return
     */
    def pivot
    (
      others: Array[FeatureLike[T]] = Array.empty,
      topK: Int = Transmogrifier.TopK,
      minSupport: Int = Transmogrifier.MinSupport,
      cleanText: Boolean = Transmogrifier.CleanText,
      trackNulls: Boolean = Transmogrifier.TrackNulls
    ): FeatureLike[OPVector] = {
      val opSetVectorizer = new OpSetVectorizer[T]()

      f.transformWith[OPVector](
        stage = opSetVectorizer.setTopK(topK).setCleanText(cleanText).setTrackNulls(trackNulls)
          .setMinSupport(minSupport),
        fs = others
      )
    }

    /**
     * Converts a sequence of OPSet features into a vector keeping the top K most common occurrences of each
     * OPSet feature (ie the final vector has length k * number of OPSet inputs). Plus an additional column
     * for "other" values - which will capture values that do not make the cut or values not seen in training
     *
     * @param others    other features to include in the pivot
     * @param topK      keep topK values
     * @param minSupport min occurances to keep a value
     * @param cleanText  if true ignores capitalization and punctuations when grouping categories
     * @param trackNulls keep a count of nulls
     * @return
     */
    def vectorize
    (
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      trackNulls: Boolean = Transmogrifier.TrackNulls,
      others: Array[FeatureLike[T]] = Array.empty
    ): FeatureLike[OPVector] =
      f.pivot(others = others, topK = topK, cleanText = cleanText, minSupport = minSupport, trackNulls = trackNulls)

  }

  /**
   * Enrichment functions for MultiPickList Feature
   *
   * @param f MultiPickList Feature
   */
  implicit class RichSetFeature(val f: FeatureLike[MultiPickList]) {

    /**
     * Apply Jaccard Similarity transformer
     *
     * @param that other MultiPickList feature
     * @return
     */
    def jaccardSimilarity(that: FeatureLike[MultiPickList]): FeatureLike[RealNN] =
      f.transformWith(new JaccardSimilarity(), that)

    /**
     * Apply N-gram Similarity transformer
     *
     * @param that other MultiPickList feature
     * @param nGramSize the size of the n-gram to be used to compute the string distance
     * @return
     */
    def toNGramSimilarity(
      that: FeatureLike[MultiPickList],
      nGramSize: Int = NGramSimilarity.nGramSize
    ): FeatureLike[RealNN] = f.transformWith(new SetNGramSimilarity(nGramSize), that)

  }

}

