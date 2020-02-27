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

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.{Impurity, OpRandomForestClassifier}
import com.salesforce.op.stages.impl.feature.{DropIndicesByTransformer, OpLDA}
import com.salesforce.op.stages.impl.preparators.MinVarianceFilter
import com.salesforce.op.stages.sparkwrappers.specific.OpEstimatorWrapper
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.feature.{IDF, IDFModel}


trait RichVectorFeature {

  /**
   * Enrichment functions for Vector Feature
   *
   * @param f FeatureLike
   */
  implicit class RichVectorFeature(val f: FeatureLike[OPVector]) {
    /**
     * Apply inverse-document frequency transformation.
     *
     * @param minDocFreq minimum number of documents in which a term should appear for filtering (default: 0)
     */
    def idf(minDocFreq: Int = 0): FeatureLike[OPVector] = {
      val idf = new IDF().setMinDocFreq(minDocFreq)
      val tr = new OpEstimatorWrapper[OPVector, OPVector, IDF, IDFModel](idf, UID[IDF])
      f.transformWith(tr)
    }

    /**
     * Apply Random Forest classifier
     *
     * @param label
     * @param maxDepth
     * @param maxBins
     * @param minInstancePerNode
     * @param minInfoGain
     * @param subSamplingRate
     * @param numTrees
     * @param impurity
     * @param seed
     * @param thresholds
     * @return
     */
    def randomForest
    (
      label: FeatureLike[RealNN],
      maxDepth: Int = 5,
      maxBins: Int = 32,
      minInstancePerNode: Int = 1,
      minInfoGain: Double = 0.0,
      subSamplingRate: Double = 1.0,
      numTrees: Int = 20,
      impurity: Impurity = Impurity.Entropy,
      seed: Long = util.Random.nextLong,
      thresholds: Array[Double] = Array.empty
    ): (FeatureLike[Prediction]) = {
      val OpRF = new OpRandomForestClassifier().setInput(label, f)
      if (thresholds.nonEmpty) OpRF.setThresholds(thresholds)

      OpRF.setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
        .setMinInstancesPerNode(minInstancePerNode)
        .setMinInfoGain(minInfoGain)
        .setSubsamplingRate(subSamplingRate)
        .setNumTrees(numTrees)
        .setImpurity(impurity.sparkName)
        .setSeed(seed)
        .getOutput()
    }

    /**
     * Apply Latent Dirichlet Allocation to compute topic distributions
     *
     * @param checkpointInterval num of iterations between two consecutive checkpoints, -1 means disabled
     * @param k                  number of topics (clusters) to infer
     * @param maxIter            maximum number of iterations
     * @param optimizer          optimizer or inference algorithm used to estimate the LDA model, "online" or "em"
     * @param subsamplingRate    fraction of the corpus to be sampled and used in mini-batch gradient descent
     * @return
     */
    def lda
    (
      checkpointInterval: Int = 10,
      k: Int = 10,
      maxIter: Int = 10,
      optimizer: String = "online",
      subsamplingRate: Double = 0.05,
      seed: Long = util.Random.nextLong()
    ): FeatureLike[OPVector] = {
      val opLDA = new OpLDA().setInput(f)
      opLDA.setCheckpointInterval(checkpointInterval)
        .setK(k).setMaxIter(maxIter)
        .setOptimizer(optimizer)
        .setSubsamplingRate(subsamplingRate)
        .setSeed(seed)
        .getOutput()
    }

    /**
     * Allows columns to be dropped from a feature vector based on properties of the
     * metadata about what is contained in each column (will work only on vectors)
     * created with [[OpVectorMetadata]]
     *
     * @param matchFn function that goes from [[OpVectorColumnMetadata]] to boolean for dropping
     *                columns (cases that evaluate to true will be dropped)
     * @return new Vector with columns removed by function
     */
    def dropIndicesBy(matchFn: OpVectorColumnMetadata => Boolean): FeatureLike[OPVector] = {
      new DropIndicesByTransformer(matchFn = matchFn).setInput(f).getOutput()
    }

    /**
     * Apply filter that removes computed features that have variance <= `minVariance``
     *
     * @param minVariance
     * @param removeBadFeatures
     * @return
     */
    def minVariance
    (
      minVariance: Double = MinVarianceFilter.MinVariance,
      removeBadFeatures: Boolean = MinVarianceFilter.RemoveBadFeatures
    ): FeatureLike[OPVector] = {
      val filter = new MinVarianceFilter()
      filter.setInput(f)
        .setMinVariance(minVariance)
        .setRemoveBadFeatures(removeBadFeatures)
        .getOutput()
    }

  }

}
