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

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.stages.impl.ModelsToTry
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
sealed trait RegressionModelsToTry extends ModelsToTry

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
private[op] trait HasLinearRegression extends Params
  with SubStage[RegressionModelSelector] {
  val sparkLR = new LinearRegression()

  final val useLR = new BooleanParam(this, "useLR",
    "boolean to decide to use LinearRegression in the model selector"
  )
  setDefault(useLR, false)

  private[op] val lRGrid = new ParamGridBuilder()


  /**
   * Linear Regression Params
   */
  private[op] def setLRParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkLR.getParam(pName).asInstanceOf[Param[T]]
    lRGrid.addGrid(p, values)
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
 * RandomForest Regressor for Model Selector
 */
private[op] trait HasRandomForestRegressor
  extends HasRandomForestBase[Regressor, SelectorRegressors] {
  override val sparkRF: Regressor = new RandomForestRegressor().asInstanceOf[Regressor]
}

/**
 * Decision Tree Regressor for Model Selector
 */
private[op] trait HasDecisionTreeRegressor
  extends HasDecisionTreeBase[Regressor, SelectorRegressors] {
  override val sparkDT: Regressor = new DecisionTreeRegressor().asInstanceOf[Regressor]
}


/**
 * GradientBoostedTree Regressor for Model Selector
 */
private[op] trait HasGradientBoostedTreeRegression
  extends HasGradientBoostedTreeBase[Regressor, SelectorRegressors] {
  override val sparkGBT: Regressor = new GBTRegressor().asInstanceOf[Regressor]
}

sealed abstract class LossType(val sparkName: String) extends EnumEntry with Serializable

object LossType extends Enum[LossType] {
  val values: Seq[LossType] = findValues

  case object Squared extends LossType("squared")
  case object Absolute extends LossType("absolute")
}


/**
 * Regressors to try in the Model Selector
 */
private[op] trait SelectorRegressors
  extends HasLinearRegression
    with HasRandomForestRegressor
    with HasDecisionTreeRegressor
    with HasGradientBoostedTreeRegression {

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

  final protected def getModelInfo: Seq[ModelInfo[Regressor]] = Seq(
    new ModelInfo(sparkLR.asInstanceOf[Regressor], lRGrid, useLR),
    new ModelInfo(sparkRF.asInstanceOf[Regressor], rFGrid, useRF),
    new ModelInfo(sparkDT.asInstanceOf[Regressor], dTGrid, useDT),
    new ModelInfo(sparkGBT.asInstanceOf[Regressor], gBTGrid, useGBT)
  )
}
