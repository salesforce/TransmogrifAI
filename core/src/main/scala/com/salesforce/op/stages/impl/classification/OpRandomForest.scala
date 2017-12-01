/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.UID
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.OpProbabilisticClassifierWrapper
import enumeratum.{Enum, EnumEntry}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

sealed abstract class Impurity(val sparkName: String) extends EnumEntry with Serializable

object Impurity extends Enum[Impurity] {
  val values: Seq[Impurity] = findValues

  case object Entropy extends Impurity("entropy")
  case object Gini extends Impurity("gini")
}


class OpRandomForest(uid: String = UID[OpRandomForest])
  extends OpProbabilisticClassifierWrapper[RandomForestClassifier, RandomForestClassificationModel](
    probClassifier = new RandomForestClassifier,
    uid = uid
  )
{

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  /**
   * Set maximum depth of the tree (>= 0).
   * E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * (default = 5)
   *
   * @group setParam
   */
  def setMaxDepth(value: Int): this.type = {
    getSparkStage.setMaxDepth(value)
    this
  }

  /**
   * Set maximum number of bins used for discretizing continuous features and for choosing how to split
   * on features at each node.  More bins give higher granularity.
   * Must be >= 2 and >= number of categories in any categorical feature.
   * (default = 32)
   *
   * @group setParam
   */
  def setMaxBins(value: Int): this.type = {
    getSparkStage.setMaxBins(value)
    this
  }

  /**
   * Set minimum number of instances each child must have after split.
   * If a split causes the left or right child to have fewer than minInstancesPerNode,
   * the split will be discarded as invalid.
   * Should be >= 1.
   * (default = 1)
   *
   * @group setParam
   */
  def setMinInstancesPerNode(value: Int): this.type = {
    getSparkStage.setMinInstancesPerNode(value)
    this
  }

  /**
   * Set minimum information gain for a split to be considered at a tree node.
   * (default = 0.0)
   *
   * @group setParam
   */
  def setMinInfoGain(value: Double): this.type = {
    getSparkStage.setMinInfoGain(value)
    this
  }

  /**
   * Set fraction of the training data used for learning each decision tree, in range (0, 1].
   * (default = 1.0)
   *
   * @group setParam
   */
  def setSubsamplingRate(value: Double): this.type = {
    getSparkStage.setSubsamplingRate(value)
    this
  }

  /**
   * Set number of trees to train (>= 1).
   * If 1, then no bootstrapping is used.  If > 1, then bootstrapping is done.
   * (default = 20)
   *
   * @group setParam
   */
  def setNumTrees(value: Int): this.type = {
    getSparkStage.setNumTrees(value)
    this
  }

  /**
   * Set criterion used for information gain calculation (case-insensitive).
   * Supported: "entropy" and "gini".
   * (default = gini)
   *
   * @group setParam
   */
  def setImpurity(value: Impurity): this.type = {
    getSparkStage.setImpurity(value.sparkName)
    this
  }

  /**
   * Set param for random seed.
   *
   * @group setParam
   */
  def setSeed(value: Long): this.type = {
    getSparkStage.setSeed(value)
    this
  }

  /**
   * Set thresholds in multiclass (or binary) classification to adjust the probability of
   * predicting each class. Array must have length equal to the number of classes, with values >= 0.
   * The class with largest value p/t is predicted, where p is the original probability of that
   * class and t is the class' threshold.
   *
   * @group setParam
   */
  def setThresholds(value: Array[Double]): this.type = {
    getSparkStage.setThresholds(value)
    this
  }
}

