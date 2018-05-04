/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml.classification

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealMap, RealNN}

import scala.reflect.runtime.universe.TypeTag

class OpRandomForestClassificationModel
(
  val treesIn: Array[DecisionTreeClassificationModel],
  numFeatures: Int,
  numClasses: Int,
  uid: String = UID[OpRandomForestClassificationModel],
  val operationName: String = "opRF"
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends RandomForestClassificationModel(uid = uid, _trees = treesIn, numFeatures = numFeatures,
  numClasses = numClasses) with OpClassifierModelBase
