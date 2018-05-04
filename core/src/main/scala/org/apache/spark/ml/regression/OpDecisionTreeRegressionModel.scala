/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml.regression

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealMap, RealNN}
import org.apache.spark.ml.tree.Node

import scala.reflect.runtime.universe.TypeTag

class OpDecisionTreeRegressionModel
(
  rootNode: Node,
  numFeatures: Int,
  uid: String = UID[OpDecisionTreeRegressionModel],
  val operationName: String = "opDTR"
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends DecisionTreeRegressionModel(uid = uid, rootNode = rootNode, numFeatures = numFeatures)
  with OpPredictionModelBase
