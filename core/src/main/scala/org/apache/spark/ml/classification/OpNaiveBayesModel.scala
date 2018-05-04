/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml.classification

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealMap, RealNN}
import org.apache.spark.ml.linalg.{Matrix, Vector}

import scala.reflect.runtime.universe.TypeTag

class OpNaiveBayesModel
(
  pi: Vector,
  theta: Matrix,
  val oldLabelsIn: Array[Double],
  val modelTypeIn: String,
  uid: String = UID[OpNaiveBayesModel],
  val operationName: String = "opNB"
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends NaiveBayesModel(uid = uid, pi = pi, theta = theta) with OpClassifierModelBase {
  this.oldLabels = oldLabelsIn
  set(modelType, modelTypeIn)
}
