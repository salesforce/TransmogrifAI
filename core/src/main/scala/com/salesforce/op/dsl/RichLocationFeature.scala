/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types.{Location, MultiPickList, OPVector, Text}

import scala.reflect.runtime.universe.TypeTag

/**
 * Enrichment functions for Location Features
 */
trait RichLocationFeature {
  self: RichTextFeature =>


  implicit class RichLocationFeature[T <: Text with Location : TypeTag](val f: FeatureLike[T])
    (implicit val ttiv: TypeTag[T#Value]) {


    /**
     * Converts a sequence of Location features into a vector keeping the top K most common occurrences of each
     * feature (ie the final vector has length k * number of inputs). Plus an additional column
     * for "other" values - which will capture values that do not make the cut or values not seen in training
     *
     * @param others    other features to include in the pivot
     * @param topK      keep topK values
     * @param minSupport    min times a value must occur to be retained in pivot
     * @param cleanText if true ignores capitalization and punctuations when grouping categories
     * @return
     */
    def vectorize
    (
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      others: Array[FeatureLike[T]] = Array.empty
    ): FeatureLike[OPVector] = {
      f.pivot(others, topK, minSupport, cleanText)
    }

  }

}
