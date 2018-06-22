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

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictionModel, OpPredictorWrapper}
import com.salesforce.op.utils.reflection.ReflectionUtils.reflectMethod
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor, OpDecisionTreeRegressorParams}

import scala.reflect.runtime.universe.TypeTag

/**
 * Wrapper for spark Decision Tree Regressor [[org.apache.spark.ml.regression.DecisionTreeRegressor]]
 * @param uid       stage uid
 */
class OpDecisionTreeRegressor(uid: String = UID[OpDecisionTreeRegressor])
  extends OpPredictorWrapper[DecisionTreeRegressor, DecisionTreeRegressionModel](
    predictor = new DecisionTreeRegressor(),
    uid = uid
  ) with OpDecisionTreeRegressorParams {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  /** @group setParam */
  override def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  override def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  override def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  override def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  override def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  override def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
   * Specifies how often to checkpoint the cached node IDs.
   * E.g. 10 means that the cache will get checkpointed every 10 iterations.
   * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
   * [[org.apache.spark.SparkContext]].
   * Must be at least 1.
   * (default = 10)
   * @group setParam
   */
  override def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  override def setImpurity(value: String): this.type = set(impurity, value)

  /** @group setParam */
  override def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setVarianceCol(value: String): this.type = set(varianceCol, value)

}

/**
 * Class that takes in a spark DecisionTreeRegressionModel and wraps it into an OP model which returns a
 * Prediction feature
 * @param sparkModel model to wrap
 * @param uid uid to give stage
 * @param operationName unique name of the operation this stage performs
 */
class OpDecisionTreeRegressionModel
(
  sparkModel: DecisionTreeRegressionModel,
  uid: String = UID[OpDecisionTreeRegressionModel],
  operationName: String = classOf[DecisionTreeRegressor].getSimpleName
)(
  implicit tti1: TypeTag[RealNN],
  tti2: TypeTag[OPVector],
  tto: TypeTag[Prediction],
  ttov: TypeTag[Prediction#Value]
) extends OpPredictionModel[DecisionTreeRegressionModel](
  sparkModel = sparkModel, uid = uid, operationName = operationName
) {
  @transient lazy val predictMirror = reflectMethod(getSparkMlStage().get, "predict")
}

