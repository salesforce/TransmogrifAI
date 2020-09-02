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

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictionModel, OpPredictorWrapper}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{OpRandomForestRegressorParams, RandomForestRegressionModel, RandomForestRegressor}
import ml.combust.mleap.core.regression.{RandomForestRegressionModel => MleapRandomForestRegressionModel}

import scala.reflect.runtime.universe.TypeTag

/**
 * Wrapper around sparj Random Forest Regressor [[org.apache.spark.ml.regression.RandomForestRegressor]]
 * @param uid       stage uid
 */
class OpRandomForestRegressor(uid: String = UID[OpRandomForestRegressor])
  extends OpPredictorWrapper[RandomForestRegressor, RandomForestRegressionModel](
    predictor = new RandomForestRegressor(),
    uid = uid
  ) with OpRandomForestRegressorParams {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  // Parameters from TreeRegressorParams:

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

  // Parameters from TreeEnsembleParams:

  /** @group setParam */
  override def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  override def setSeed(value: Long): this.type = set(seed, value)

  // Parameters from RandomForestParams:

  /** @group setParam */
  override def setNumTrees(value: Int): this.type = set(numTrees, value)

  /** @group setParam */
  override def setFeatureSubsetStrategy(value: String): this.type =
  set(featureSubsetStrategy, value)

}

/**
 * Class that takes in a spark RandomForestRegressionModel and wraps it into an OP model which returns a
 * Prediction feature
 * @param sparkModel model to wrap
 * @param uid uid to give stage
 * @param operationName unique name of the operation this stage performs
 */
class OpRandomForestRegressionModel
(
  sparkModel: RandomForestRegressionModel,
  uid: String = UID[OpRandomForestRegressionModel],
  operationName: String = classOf[RandomForestRegressor].getSimpleName
)(
  implicit tti1: TypeTag[RealNN],
  tti2: TypeTag[OPVector],
  tto: TypeTag[Prediction],
  ttov: TypeTag[Prediction#Value]
) extends OpPredictionModel[RandomForestRegressionModel](
  sparkModel = sparkModel, uid = uid, operationName = operationName
)

