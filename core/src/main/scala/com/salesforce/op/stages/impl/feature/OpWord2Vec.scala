/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.sparkwrappers.specific.OpEstimatorWrapper
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}

/**
 * Wrapper around spark ml word2vec for use with OP pipelines
 */
class OpWord2Vec(uid: String = UID[OpWord2Vec])
  extends OpEstimatorWrapper[TextList, OPVector, Word2Vec, Word2VecModel](estimator = new Word2Vec(), uid = uid) {
  /**
   * Number of iterations
   * Default is 1
   *
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = {
    getSparkMlStage().get.setMaxIter(value)
    this
  }

  /**
   * Maximum length (in words) of each sentence in the input data.
   * Any sentence longer than this threshold will be divided into chunks of
   * up to `maxSentenceLength` size.
   * Default: 1000
   *
   * @group param
   */
  def setMaxSentenceLength(value: Int): this.type = {
    getSparkMlStage().get.setMaxSentenceLength(value)
    this
  }

  /**
   * The minimum number of times a token must appear to be included in the word2vec model's
   * vocabulary.
   * Default: 5
   *
   * @group param
   */
  def setMinCount(value: Int): this.type = {
    getSparkMlStage().get.setMinCount(value)
    this
  }

  /**
   * Number of partitions for sentences of words
   * Default: 1
   *
   * @group param
   */
  def setNumPartitions(value: Int): this.type = {
    getSparkMlStage().get.setNumPartitions(value)
    this
  }

  def setSeed(value: Long): this.type = {
    getSparkMlStage().get.setSeed(value)
    this
  }

  /**
   * The window size (context words from [-window, window])
   * Default: 5
   *
   * @group expertParam
   */
  def setWindowSize(value: Int): this.type = {
    getSparkMlStage().get.setWindowSize(value)
    this
  }

  /**
   * The dimension of the code that you want to transform from words
   * Default: 100
   *
   * @group param
   */
  def setVectorSize(value: Int): this.type = {
    getSparkMlStage().get.setVectorSize(value)
    this
  }

  /**
   * Initial learning rate
   * Default: 0.025
   *
   * @group param
   */
  def setStepSize(value: Double): this.type = {
    getSparkMlStage().get.setStepSize(value)
    this
  }

}
