/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
     * @return vector feature
     */
    def transmogrify(): FeatureLike[OPVector] =
      Transmogrifier.transmogrify(features.toSeq)(TransmogrifierDefaults).combine()

    /**
     * Convert features into a single vector feature using the feature engineering steps most likely to provide
     * good results based on the types of the individual features passed in.
     *
     * @return vector feature
     */
    def autoTransform(): FeatureLike[OPVector] = transmogrify()

  }

}
