/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml.regression

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealMap, RealNN}
import org.apache.spark.ml.linalg.Vector

import scala.reflect.runtime.universe.TypeTag

class OpLinearPredictionModel
(
  coefficients: Vector,
  intercept: Double,
  uid: String = UID[OpLinearPredictionModel],
  val operationName: String = "opLP"
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends LinearRegressionModel(uid = uid, coefficients = coefficients, intercept = intercept)
  with OpPredictionModelBase
