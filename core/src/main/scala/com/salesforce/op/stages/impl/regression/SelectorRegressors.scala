/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.stages.impl.regression.RegressorType._
import com.salesforce.op.stages.impl.selector._
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.ml.regression._
import org.apache.spark.ml.tuning.ParamGridBuilder
import enumeratum._

import scala.reflect.ClassTag


/**
 * Enumeration of possible regression models in Model Selector
 */
sealed trait RegressionModelsToTry extends EnumEntry with Serializable

object RegressionModelsToTry extends Enum[RegressionModelsToTry] {
  val values = findValues
  case object LinearRegression extends RegressionModelsToTry
  case object DecisionTreeRegression extends RegressionModelsToTry
  case object RandomForestRegression extends RegressionModelsToTry
  case object GBTRegression extends RegressionModelsToTry
}


sealed abstract class Solver(val sparkName: String) extends EnumEntry with Serializable

object Solver extends Enum[Solver] {
  val values: Seq[Solver] = findValues

  case object LBFGS extends Solver("l-bfgs")
  case object Normal extends Solver("normal")
  case object Auto extends Solver("auto")
}

/**
 * Linear Regression for Model Selector
 */
private[regression] trait HasLinearRegression extends Params
  with SubStage[RegressionModelSelector] {
  val sparkLR = new LinearRegression()

  final val useLR = new BooleanParam(this, "useLR",
    "boolean to decide to use LinearRegression in the model selector"
  )
  setDefault(useLR, true)

  private[regression] val lRGrid = new ParamGridBuilder()


  /**
   * Linear Regression Params
   */
  private[regression] def setLRParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkLR.getParam(pName).asInstanceOf[Param[T]]
    if (values.distinct.length == 1) sparkLR.set(p, values.head)
    else lRGrid.addGrid(p, values)
    subStage.foreach(_.setLRParams[T](pName, values))
    this
  }

  /**
   * Set the regularization parameter.
   * Default is 0.0.
   *
   * @group setParam
   */
  def setLinearRegressionRegParam(value: Double*): this.type = setLRParams("regParam", value)

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
   * For 0 < alpha < 1, the penalty is a combination of L1 and L2.
   * Default is 0.0 which is an L2 penalty.
   *
   * @group setParam
   */
  def setLinearRegressionElasticNetParam(value: Double*): this.type = setLRParams("elasticNetParam", value)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  def setLinearRegressionMaxIter(value: Int*): this.type = setLRParams("maxIter", value)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  def setLinearRegressionTol(value: Double*): this.type = setLRParams("tol", value)

  /**
   * Whether to fit an intercept term.
   * Default is true.
   *
   * @group setParam
   */
  def setLinearRegressionFitIntercept(value: Boolean*): this.type = setLRParams("fitIntercept", value)

  /**
   * Whether to standardize the training features before fitting the model.
   * The coefficients of models will be always returned on the original scale,
   * so it will be transparent for users. Note that with/without standardization,
   * the models should be always converged to the same solution when no regularization
   * is applied. In R's GLMNET package, the default behavior is true as well.
   * Default is true.
   *
   * @group setParam
   */
  def setLinearRegressionStandardization(value: Boolean*): this.type = setLRParams("standardization", value)

  /**
   * Set the solver algorithm used for optimization. In case of linear regression, this can be "l-bfgs", "normal" and
   * "auto".
   * "l-bfgs": Limited-memory BFGS which is a limited-memory quasi-Newton optimization method.
   * "normal": Normal Equation as an analytical solution to the linear regression problem.
   * "auto" (default): solver algorithm is selected automatically. The Normal Equations solver will be used when
   * possible, but this will automatically fall back to iterative optimization methods when needed.
   *
   * @group setParam
   */
  def setLinearRegressionSolver(value: Solver*): this.type = setLRParams("solver", value.map(_.sparkName))

}

/**
 * Decision Tree Regressor for Model Selector
 */
private[regression] trait HasRandomForestRegressor
  extends HasRandomForestBase[Regressor, SelectorRegressors] {
  override val sparkRF: Regressor = new RandomForestRegressor().asInstanceOf[Regressor]
}

/**
 * Decision Tree Regressor for Model Selector
 */
private[regression] trait HasDecisionTreeRegressor
  extends HasDecisionTreeBase[Regressor, SelectorRegressors] {
  override val sparkDT: Regressor = new DecisionTreeRegressor().asInstanceOf[Regressor]
}


sealed abstract class LossType(val sparkName: String) extends EnumEntry with Serializable

object LossType extends Enum[LossType] {
  val values: Seq[LossType] = findValues

  case object Squared extends LossType("squared")
  case object Absolute extends LossType("absolute")
}

/**
 * Gradient Boosted Tree Regressor for Model Selector
 */
private[regression] trait HasGradientBoostedTreeRegression extends Params
  with SubStage[RegressionModelSelector] {

  val sparkGBT = new GBTRegressor()

  final val useGBT = new BooleanParam(this, "useGBT",
    "boolean to decide to use GradientBoostedTree in the model selector"
  )
  setDefault(useGBT, true)

  private[impl] val gBTGrid = new ParamGridBuilder()

  /**
   * Gradient Boosted Tree Params
   */

  private[regression] def setGBTParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkGBT.getParam(pName).asInstanceOf[Param[T]]
    if (values.distinct.length == 1) sparkGBT.set(p, values.head)
    else gBTGrid.addGrid(p, values)
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
}

/**
 * Regressors to try in the Model Selector
 */
private[impl] trait SelectorRegressors
  extends HasLinearRegression
    with HasRandomForestRegressor
    with HasDecisionTreeRegressor
    with HasGradientBoostedTreeRegression
    with SelectorModels[Regressor, RegressionModelSelector]
    with Stage1ParamNamesBase {

  // scalastyle:off
  import RegressionModelsToTry._

  // scalastyle:on

  /**
   * Set the models to try for the model selector.
   * The models can be LinearRegression, RandomForestRegressor, DecisionTreeRegressor or GradientBoostedTreeRegressor
   *
   * @group setParam
   */
  def setModelsToTry(modelsToTry: RegressionModelsToTry*): this.type = {
    val potentialModelSet = modelsToTry.toSet
    set(useLR, potentialModelSet(LinearRegression))
    set(useRF, potentialModelSet(RandomForestRegression))
    set(useDT, potentialModelSet(DecisionTreeRegression))
    set(useGBT, potentialModelSet(GBTRegression))
    subStage.foreach(_.setModelsToTry(modelsToTry: _*))
    this
  }

  final override protected[impl] def modelInfo: Seq[ModelInfo[Regressor]] =
    Seq(ModelInfo(sparkLR.asInstanceOf[Regressor], lRGrid, useLR),
      ModelInfo(sparkRF.asInstanceOf[Regressor], rFGrid, useRF),
      ModelInfo(sparkDT.asInstanceOf[Regressor], dTGrid, useDT),
      ModelInfo(sparkGBT.asInstanceOf[Regressor], gBTGrid, useGBT))
}
