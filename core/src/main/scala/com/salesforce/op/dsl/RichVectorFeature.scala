/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.dsl

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.{Impurity, OpRandomForest}
import com.salesforce.op.stages.impl.feature.OpLDA
import com.salesforce.op.stages.sparkwrappers.specific.OpEstimatorWrapper
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
    def randomForest(
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
    ): (FeatureLike[RealNN], FeatureLike[OPVector], FeatureLike[OPVector]) = {
      val OpRF = new OpRandomForest().setInput(label, f)
      if (thresholds.nonEmpty) OpRF.setThresholds(thresholds)

      OpRF.setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
        .setMinInstancesPerNode(minInstancePerNode)
        .setMinInfoGain(minInfoGain)
        .setSubsamplingRate(subSamplingRate)
        .setNumTrees(numTrees)
        .setImpurity(impurity)
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
  }

}
