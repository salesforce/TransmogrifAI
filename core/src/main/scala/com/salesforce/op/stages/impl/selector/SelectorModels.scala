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

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.regression.LossType
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.ml.tuning.ParamGridBuilder

import scala.reflect.ClassTag

case class ModelInfo[E <: Estimator[_]](sparkEstimator: E, grid: ParamGridBuilder, useModel: BooleanParam) {
  def modelName: String = sparkEstimator.getClass.getSimpleName
}

/**
 * Input and Output Param names for Selectors containing models extending the Predictor and ClassifierPredictor spark
 * classes
 */
private[op] object StageParamNames {
  val inputParam1Name = "labelCol"
  val inputParam2Name = "featuresCol"

  val outputParam1Name = "predictionCol"
  val outputParam2Name = "rawPredictionCol"
  val outputParam3Name = "probabilityCol"

  val stage1OperationName = "label_prediction"
  val stage2OperationName = "label_rawPrediction"
  val stage3OperationName = "label_probability"
}

private[op] trait StageOperationName {
  val stage1OperationName = StageParamNames.stage1OperationName
  val stage2OperationName = StageParamNames.stage2OperationName
  val stage3OperationName = StageParamNames.stage3OperationName
}

/**
 * For passing params to substage
 *
 * @tparam MS model selector
 */
private[op] trait SubStage[+MS <: SubStage[MS]] {
  protected def subStage: Option[MS] = None
}

/**
 * Random Forest for Model Selector
 */
private[op] trait HasRandomForestBase[E <: Estimator[_], +MS <: HasRandomForestBase[E, MS]]
  extends Params with SubStage[MS] {

  val sparkRF: E

  final val useRF = new BooleanParam(this, "useRF",
    "boolean to decide to use RandomForestRegressor in the model selector"
  )
  setDefault(useRF, false)

  private[op] val rFGrid = new ParamGridBuilder()


  /**
   * Random Forest Regressor Params
   */

  private[op] def setRFParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkRF.getParam(pName).asInstanceOf[Param[T]]
    rFGrid.addGrid(p, values)
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
private[op] trait HasDecisionTreeBase[E <: Estimator[_], +MS <: HasDecisionTreeBase[E, MS]]
  extends Params with SubStage[MS] {

  val sparkDT: E

  final val useDT = new BooleanParam(this, "useDT",
    "boolean to decide to use DecisionTree in the model selector"
  )
  setDefault(useDT, false)

  private[op] val dTGrid = new ParamGridBuilder()

  /**
   * Decision Tree Params
   */
  private[op] def setDTParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkDT.getParam(pName).asInstanceOf[Param[T]]
    dTGrid.addGrid(p, values)
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
 * Gradient Boosted Tree Regressor for Model Selector
 */
private[op] trait HasGradientBoostedTreeBase[E <: Estimator[_], +MS <: HasGradientBoostedTreeBase[E, MS]]
  extends Params with SubStage[MS] {

  val sparkGBT: E

  final val useGBT = new BooleanParam(this, "useGBT",
    "boolean to decide to use GradientBoostedTree in the model selector"
  )
  setDefault(useGBT, false)

  private[op] val gBTGrid = new ParamGridBuilder()

  /**
   * Gradient Boosted Tree Params
   */
  private[op] def setGBTParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkGBT.getParam(pName).asInstanceOf[Param[T]]
    gBTGrid.addGrid(p, values)
    subStage.foreach(_.setGBTParams[T](pName, values))
    this
  }

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  def setGradientBoostedTreeMaxIter(value: Int*): this.type = setGBTParams("maxIter", value)

  /**
   * Set maximum depth of the tree (>= 0).
   * E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * (default = 5)
   *
   * @group setParam
   */
  def setGradientBoostedTreeMaxDepth(value: Int*): this.type = setGBTParams("maxDepth", value)

  /**
   * Set maximum number of bins used for discretizing continuous features and for choosing how to split
   * on features at each node.  More bins give higher granularity.
   * Must be >= 2 and >= number of categories in any categorical feature.
   * (default = 32)
   *
   * @group setParam
   */
  def setGradientBoostedTreeMaxBins(value: Int*): this.type = setGBTParams("maxBins", value)

  /**
   * Set minimum number of instances each child must have after split.
   * If a split causes the left or right child to have fewer than minInstancesPerNode,
   * the split will be discarded as invalid.
   * Should be >= 1.
   * (default = 1)
   *
   * @group setParam
   */
  def setGradientBoostedTreeMinInstancesPerNode(value: Int*): this.type = setGBTParams("minInstancesPerNode", value)

  /**
   * Set minimum information gain for a split to be considered at a tree node.
   * (default = 0.0)
   *
   * @group setParam
   */
  def setGradientBoostedTreeMinInfoGain(value: Double*): this.type = setGBTParams("minInfoGain", value)

  /**
   * Set loss function which GBT tries to minimize.
   * Supported: "squared" (L2) and "absolute" (L1)
   * (default = squared)
   *
   * @group setParam
   */
  def setGradientBoostedTreeLossType(value: LossType*): this.type = setGBTParams("lossType", value.map(_.sparkName))

  /**
   * Set param for random seed.
   *
   * @group setParam
   */
  def setGradientBoostedTreeSeed(value: Long*): this.type = setGBTParams("seed", value)

  /**
   * Set param for Step size (a.k.a learning rate) in interval [0, 1] for shrinking the contribution of each estimator
   * (default = 0.1)
   *
   * @group setParam
   */
  def setGradientBoostedTreeStepSize(value: Double*): this.type = setGBTParams("stepSize", value)


  /**
   * Set fraction of the training data used for learning each decision tree, in range (0, 1].
   * (default = 1.0)
   *
   * @group setParam
   */
  def setGradientBoostedTreeSubsamplingRate(value: Double*): this.type = setGBTParams("subsamplingRate", value)

  /**
   * Set criterion used for information gain calculation (case-insensitive).
   * Supported: "entropy" and "gini".
   * (default = gini)
   *
   * @group setParam
   */
  def setGradientBoostedTreeImpurity(value: Impurity*): this.type = setGBTParams("impurity", value.map(_.sparkName))

}


