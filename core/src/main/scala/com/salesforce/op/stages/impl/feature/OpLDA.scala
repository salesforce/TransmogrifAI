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
import com.salesforce.op.stages.sparkwrappers.specific.OpEstimatorWrapper
import org.apache.spark.ml.clustering.{LDA, LDAModel}

/**
 * Wrapper around spark ml LDA (Latent Dirichlet Allocation) for use with OP pipelines
 */
class OpLDA(uid: String = UID[OpLDA])
  extends OpEstimatorWrapper[OPVector, OPVector, LDA, LDAModel](estimator = new LDA(), uid = uid) {

  override val inputParamName: String = "featuresCol"
  override val outputParamName: String = "topicDistributionCol"

  /**
   * Set param for checkpoint interval (>= 1) or disable checkpoint (-1).
   * E.g. 10 means that the cache will get checkpointed every 10 iterations.
   *
   * @group setParam
   */
  def setCheckpointInterval(value: Int): this.type = {
    getSparkMlStage().get.setCheckpointInterval(value)
    this
  }

  /**
   * Set param for concentration parameter (commonly named "alpha") for the prior placed on documents'
   * distributions over topics ("theta").
   *
   * This is the parameter to a Dirichlet distribution, where larger values mean more smoothing (more regularization).
   *
   * If not set by the user, then docConcentration is set automatically. If set to
   * singleton vector [alpha], then alpha is replicated to a vector of length k in fitting.
   * Otherwise, the docConcentration vector must be length k.
   * (default = automatic)
   *
   * Optimizer-specific parameter settings:
   *  - EM
   *     - Currently only supports symmetric distributions, so all values in the vector should be
   *       the same.
   *     - Values should be > 1.0
   *     - default = uniformly (50 / k) + 1, where 50/k is common in LDA libraries and +1 follows
   *       from Asuncion et al. (2009), who recommend a +1 adjustment for EM.
   *  - Online
   *     - Values should be >= 0
   *     - default = uniformly (1.0 / k), following the implementation from
   *       [[https://github.com/Blei-Lab/onlineldavb]].
   *
   * @group setParam
   */
  def setDocConcentation(value: Array[Double]): this.type = {
    getSparkMlStage().get.setDocConcentration(value)
    this
  }

  def setDocConcentration(value: Double): this.type = {
    getSparkMlStage().get.setDocConcentration(value)
    this
  }

  /**
   * Set param for number of topics (clusters) to infer. Must be > 1.
   * Default: 10.
   *
   * @group setParam
   */
  def setK(value: Int): this.type = {
    getSparkMlStage().get.setK(value)
    this
  }

  /**
   * Set param for maximum number of iterations (>= 0).
   * Default: 20
   *
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = {
    getSparkMlStage().get.setMaxIter(value)
    this
  }

  /**
   * Set param for optimizer or inference algorithm used to estimate the LDA model.
   *
   * Currently supported (case-insensitive):
   *  - "online": Online Variational Bayes (default)
   *  - "em": Expectation-Maximization
   *
   * For details, see the following papers:
   *  - Online LDA:
   *     Hoffman, Blei and Bach.  "Online Learning for Latent Dirichlet Allocation."
   *     Neural Information Processing Systems, 2010.
   *     [[http://www.cs.columbia.edu/~blei/papers/HoffmanBleiBach2010b.pdf]]
   *  - EM:
   *     Asuncion et al.  "On Smoothing and Inference for Topic Models."
   *     Uncertainty in Artificial Intelligence, 2009.
   *     [[http://arxiv.org/pdf/1205.2662.pdf]]
   *
   * @group setParam
   */
  def setOptimizer(value: String): this.type = {
    getSparkMlStage().get.setOptimizer(value)
    this
  }

  /**
   * Set param for random seed.
   *
   * @group setParam
   */
  def setSeed(value: Long): this.type = {
    getSparkMlStage().get.setSeed(value)
    this
  }

  /**
   * For Online optimizer only: optimizer = "online".
   *
   * Set param for fraction of the corpus to be sampled and used in each iteration of mini-batch gradient descent,
   * in range (0, 1].
   *
   * Note that this should be adjusted in synch with [[LDA.maxIter]]
   * so the entire corpus is used.  Specifically, set both so that
   * maxIterations * miniBatchFraction >= 1.
   *
   * Note: This is the same as the `miniBatchFraction` parameter in
   *       [[org.apache.spark.mllib.clustering.OnlineLDAOptimizer]].
   *
   * Default: 0.05, i.e., 5% of total documents.
   *
   * @group setParam
   */
  def setSubsamplingRate(value: Double): this.type = {
    getSparkMlStage().get.setSubsamplingRate(value)
    this
  }

  /**
   * Set param for concentration parameter (commonly named "beta" or "eta") for the prior placed on topics'
   * distributions over terms.
   *
   * This is the parameter to a symmetric Dirichlet distribution.
   *
   * Note: The topics' distributions over terms are called "beta" in the original LDA paper
   * by Blei et al., but are called "phi" in many later papers such as Asuncion et al., 2009.
   *
   * If not set by the user, then topicConcentration is set automatically.
   *  (default = automatic)
   *
   * Optimizer-specific parameter settings:
   *  - EM
   *     - Value should be > 1.0
   *     - default = 0.1 + 1, where 0.1 gives a small amount of smoothing and +1 follows
   *       Asuncion et al. (2009), who recommend a +1 adjustment for EM.
   *  - Online
   *     - Value should be >= 0
   *     - default = (1.0 / k), following the implementation from
   *       [[https://github.com/Blei-Lab/onlineldavb]].
   *
   * @group setParam
   */
  def setTopicConcentration(value: Double): this.type = {
    getSparkMlStage().get.setTopicConcentration(value)
    this
  }
}
