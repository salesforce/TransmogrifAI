/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */
package org.apache.spark.ml.regression

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealMap, RealNN}

import scala.reflect.runtime.universe.TypeTag


class OpRandomForestRegressionModel
(
  val treesIn: Array[DecisionTreeRegressionModel],
  numFeatures: Int,
  uid: String = UID[OpRandomForestRegressionModel],
  val operationName: String = "opRFR"
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends RandomForestRegressionModel(uid = uid, _trees = treesIn, numFeatures = numFeatures)
  with OpPredictionModelBase
