/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.stages.impl.classification._
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.ml.tuning.ParamGridBuilder

import scala.reflect.ClassTag

case class ModelInfo[E <: Estimator[_]](sparkEstimator: E, grid: ParamGridBuilder, use: BooleanParam)

/**
 * Input and Output Param names for Selectors having one stage
 */
private[impl] trait Stage1ParamNamesBase {
  val inputParam1Name = "labelCol"
  val inputParam2Name = "featuresCol"
  val outputParam1Name = "predictionCol"
  val stage1OperationName = "label_prediction"
}

/**
 * Input and Output Param names for Selectors having three stages
 */
private[impl] trait Stage3ParamNamesBase extends Stage1ParamNamesBase {
  val outputParam2Name = "rawPredictionCol"
  val outputParam3Name = "probabilityCol"

  val stage2OperationName = "label_rawPrediction"
  val stage3OperationName = "label_probability"
}

/**
 * For passing params to substage
 *
 * @tparam MS model selector
 */
private[impl] trait SubStage[+MS <: SubStage[MS]] {
  protected def subStage: Option[MS] = None
}

/**
 * Random Forest for Model Selector
 */
private[impl] trait HasRandomForestBase[E <: Estimator[_], +MS <: HasRandomForestBase[E, MS]]
  extends Params with SubStage[MS] {

  val sparkRF: E

  final val useRF = new BooleanParam(this, "useRF",
    "boolean to decide to use RandomForestRegressor in the model selector"
  )
  setDefault(useRF, true)

  private[impl] val rFGrid = new ParamGridBuilder()


  /**
   * Random Forest Regressor Params
   */

  private[op] def setRFParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkRF.getParam(pName).asInstanceOf[Param[T]]
    if (values.distinct.length == 1) sparkRF.set(p, values.head)
    else rFGrid.addGrid(p, values)
    subStage.foreach(_.setRFParams[T](pName, values))
    this
  }


  /**
   * Set maximum depth of the tree (>= 0).
   * E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * (default = 5)
   *
   * @group setParam
   */
  def setRandomForestMaxDepth(value: Int*): this.type = setRFParams("maxDepth", value)

  /**
   * Set maximum number of bins used for discretizing continuous features and for choosing how to split
   * on features at each node.  More bins give higher granularity.
   * Must be >= 2 and >= number of categories in any categorical feature.
   * (default = 32)
   *
   * @group setParam
   */
  def setRandomForestMaxBins(value: Int*): this.type = setRFParams("maxBins", value)

  /**
   * Set minimum number of instances each child must have after split.
   * If a split causes the left or right child to have fewer than minInstancesPerNode,
   * the split will be discarded as invalid.
   * Should be >= 1.
   * (default = 1)
   *
   * @group setParam
   */
  def setRandomForestMinInstancesPerNode(value: Int*): this.type = setRFParams("minInstancesPerNode", value)

  /**
   * Set minimum information gain for a split to be considered at a tree node.
   * (default = 0.0)
   *
   * @group setParam
   */
  def setRandomForestMinInfoGain(value: Double*): this.type = setRFParams("minInfoGain", value)

  /**
   * Set fraction of the training data used for learning each decision tree, in range (0, 1].
   * (default = 1.0)
   *
   * @group setParam
   */
  def setRandomForestSubsamplingRate(value: Double*): this.type = setRFParams("subsamplingRate", value)

  /**
   * Set number of trees to train (>= 1).
   * If 1, then no bootstrapping is used.  If > 1, then bootstrapping is done.
   * (default = 20)
   *
   * @group setParam
   */
  def setRandomForestNumTrees(value: Int*): this.type = setRFParams("numTrees", value)

  /**
   * Set criterion used for information gain calculation (case-insensitive).
   * Supported: "entropy" and "gini".
   * (default = gini)
   *
   * @group setParam
   */
  def setRandomForestImpurity(value: Impurity*): this.type = setRFParams("impurity", value.map(_.sparkName))

  /**
   * Set param for random seed.
   *
   * @group setParam
   */
  def setRandomForestSeed(value: Long*): this.type = setRFParams("seed", value)
}

/**
 * Decision Tree for Model Selector
 */
private[impl] trait HasDecisionTreeBase[E <: Estimator[_], +MS <: HasDecisionTreeBase[E, MS]]
  extends Params with SubStage[MS] {

  val sparkDT: E

  final val useDT = new BooleanParam(this, "useDT",
    "boolean to decide to use DecisionTree in the model selector"
  )
  setDefault(useDT, true)

  private[impl] val dTGrid = new ParamGridBuilder()

  /**
   * Decision Tree Params
   */
  private[op] def setDTParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkDT.getParam(pName).asInstanceOf[Param[T]]
    if (values.distinct.length == 1) sparkDT.set(p, values.head)
    else dTGrid.addGrid(p, values)
    subStage.foreach(_.setDTParams[T](pName, values))
    this
  }

  /**
   * Set maximum depth of the tree (>= 0).
   * E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * (default = 5)
   *
   * @group setParam
   */
  def setDecisionTreeMaxDepth(value: Int*): this.type = setDTParams("maxDepth", value)

  /**
   * Set maximum number of bins used for discretizing continuous features and for choosing how to split
   * on features at each node.  More bins give higher granularity.
   * Must be >= 2 and >= number of categories in any categorical feature.
   * (default = 32)
   *
   * @group setParam
   */
  def setDecisionTreeMaxBins(value: Int*): this.type = setDTParams("maxBins", value)

  /**
   * Set minimum number of instances each child must have after split.
   * If a split causes the left or right child to have fewer than minInstancesPerNode,
   * the split will be discarded as invalid.
   * Should be >= 1.
   * (default = 1)
   *
   * @group setParam
   */
  def setDecisionTreeMinInstancesPerNode(value: Int*): this.type = setDTParams("minInstancesPerNode", value)

  /**
   * Set minimum information gain for a split to be considered at a tree node.
   * (default = 0.0)
   *
   * @group setParam
   */
  def setDecisionTreeMinInfoGain(value: Double*): this.type = setDTParams("minInfoGain", value)

  /**
   * Set criterion used for information gain calculation (case-insensitive).
   * Supported: "entropy" and "gini".
   * (default = gini)
   *
   * @group setParam
   */
  def setDecisionTreeImpurity(value: Impurity*): this.type = setDTParams("impurity", value.map(_.sparkName))

  /**
   * Set param for random seed.
   *
   * @group setParam
   */
  def setDecisionTreeSeed(value: Long*): this.type = setDTParams("seed", value)
}


/**
 * Models used for the model Selectoe
 *
 * @tparam E  type of Models
 * @tparam MS model selector
 */
private[impl] trait SelectorModels[E <: Estimator[_], +MS <: SelectorModels[E, MS]] extends SubStage[MS] {
  protected[impl] def modelInfo: Seq[ModelInfo[E]]
}
