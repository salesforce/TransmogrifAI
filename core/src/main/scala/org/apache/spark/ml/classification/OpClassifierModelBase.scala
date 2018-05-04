/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml.classification

import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.base.binary.OpTransformer2
import org.apache.spark.ml.linalg.Vector


trait OpClassifierModelBase extends OpTransformer2[RealNN, OPVector, Prediction] {

  self: ProbabilisticClassificationModel[Vector, _] =>


  /**
   * Function used to convert input to output
   */
  override def transformFn: (RealNN, OPVector) => Prediction = (label, features) => {
    val raw = predictRaw(features.value)
    val prob = raw2probability(raw)
    val pred = probability2prediction(prob)

    Prediction(rawPrediction = raw, probability = prob, prediction = pred)
  }

}
