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

import com.salesforce.op.features.types._
import com.salesforce.op.features.{FeatureLike, OPFeature}
import com.salesforce.op.stages.impl.feature._


trait RichFeaturesCollection {

  /**
   * Enrichment functions for a collection of vector features
   *
   * @param features a collection of vector features
   */
  implicit class RichVectorFeaturesCollection(val features: TraversableOnce[FeatureLike[OPVector]]) {

    /**
     * Combine a collection of vector features into a single vector feature
     *
     * @return a feature of type vector
     */
    def combine(): FeatureLike[OPVector] = new VectorsCombiner().setInput(features.toArray).getOutput()
  }

  /**
   * Enrichment functions for a collection of arbitrary features
   *
   * @param features a collection of arbitrary features
   */
  implicit class RichAnyFeaturesCollection(val features: TraversableOnce[OPFeature]) {

    /**
     * Convert features into a single vector feature using the feature engineering steps most likely to provide
     * good results based on the types of the individual features passed in
     *
     * @param label optional label feature to be passed into stages that require the label column
     * @return vector feature
     */
    def transmogrify(label: Option[FeatureLike[RealNN]] = None): FeatureLike[OPVector] =
      Transmogrifier.transmogrify(features = features.toSeq, label = label)(TransmogrifierDefaults).combine()

    /**
     * Convert features into a single vector feature using the feature engineering steps most likely to provide
     * good results based on the types of the individual features passed in
     *
     * @param label optional label feature to be passed into stages that require the label column
     * @return vector feature
     */
    def autoTransform(label: Option[FeatureLike[RealNN]] = None): FeatureLike[OPVector] = transmogrify(label = label)

  }

}
