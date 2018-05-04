/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */
package org.apache.spark.ml.regression

import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.base.binary.OpTransformer2
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector

trait OpPredictionModelBase extends OpTransformer2[RealNN, OPVector, Prediction] {
  self: PredictionModel[Vector, _] =>

  /**
   * Function used to convert input to output
   */
  override def transformFn: (RealNN, OPVector) => Prediction = (label, features) =>
    Prediction(prediction = predict(features.value))

}
