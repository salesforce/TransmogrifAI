/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.Vector


private[op] object ProbabilisticClassifierType {
  type ProbClassifierModel = ProbabilisticClassificationModel[Vector, _]

  type ProbClassifier = ProbabilisticClassifier[Vector,
    _ <: ProbabilisticClassifier[Vector, _, _],
    _ <: ProbClassifierModel]
}
