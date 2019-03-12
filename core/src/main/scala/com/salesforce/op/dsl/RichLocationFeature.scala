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

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types.{Location, OPVector, Text}
import com.salesforce.op.stages.impl.feature.OpOneHotVectorizer

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
     * @param maxPctCardinality Max percentage of distinct values a categorical feature can have (between 0.0 and 1.00)
     *
     * @return
     */
    def vectorize
    (
      topK: Int,
      minSupport: Int,
      cleanText: Boolean,
      others: Array[FeatureLike[T]] = Array.empty,
      maxPctCardinality: Double = OpOneHotVectorizer.MaxPctCardinality
    ): FeatureLike[OPVector] = {
      f.pivot(others, topK, minSupport, cleanText, maxPctCardinality = maxPctCardinality)
    }

  }

}
