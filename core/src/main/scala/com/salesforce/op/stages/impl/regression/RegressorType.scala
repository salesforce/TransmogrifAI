/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.regression


import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.RegressionModel

object RegressorType {
  type RegressorModel = RegressionModel[Vector, _]
  type Regressor = Estimator[_ <: RegressorModel]
}
