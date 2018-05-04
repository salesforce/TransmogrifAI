/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml.classification

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import org.apache.spark.ml.linalg.{Matrix, Vector}

import scala.reflect.runtime.universe.TypeTag

class OpLogisticRegressionModel
(
  coefficientMatrix: Matrix,
  interceptVector: Vector,
  numClasses: Int,
  val isMultinomial: Boolean,
  val operationName: String = "opLR",
  uid: String = UID[OpLogisticRegressionModel]
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends LogisticRegressionModel(uid = uid, coefficientMatrix = coefficientMatrix,
  interceptVector = interceptVector, numClasses = numClasses, isMultinomial = isMultinomial) with OpClassifierModelBase
