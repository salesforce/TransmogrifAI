/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml.regression

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealMap, RealNN}

import scala.reflect.runtime.universe.TypeTag

class OpGBTRegressionModel
(
  val treesIn: Array[DecisionTreeRegressionModel],
  val treeWeightsIn: Array[Double],
  numFeatures: Int,
  uid: String = UID[OpGBTRegressionModel],
  val operationName: String = "opGBTR"
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends GBTRegressionModel(uid = uid, _trees = treesIn, _treeWeights = treeWeightsIn, numFeatures = numFeatures)
  with OpPredictionModelBase
