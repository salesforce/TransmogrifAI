/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml.classification

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealMap, RealNN}
import org.apache.spark.ml.tree.Node

import scala.reflect.runtime.universe.TypeTag

class OpDecisionTreeClassificationModel
(
  rootNode: Node,
  numFeatures: Int,
  numClasses: Int,
  uid: String = UID[OpDecisionTreeClassificationModel],
  val operationName: String = "opDTC"
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends DecisionTreeClassificationModel(uid = uid, rootNode = rootNode, numFeatures = numFeatures,
  numClasses = numClasses) with OpClassifierModelBase
