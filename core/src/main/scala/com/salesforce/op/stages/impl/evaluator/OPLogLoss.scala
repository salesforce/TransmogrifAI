/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.evaluator

import com.salesforce.op.evaluators.{Evaluators, OpBinaryClassificationEvaluatorBase, OpMultiClassificationEvaluatorBase, SingleMetric}
import com.twitter.algebird.AveragedValue
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset
import com.salesforce.op.utils.spark.RichDataset.RichDataset

/**
 * Logarithmic Loss metric, implemented as both Binary and MultiClass evaluators
 */
object LogLoss {

  def logLossFun(ds: Dataset[(Double, Vector, Vector, Double)]): Double = {
    import ds.sparkSession.implicits._
    require(!ds.isEmpty, "Dataset is empty, log loss cannot be calculated")
    val avg = ds.map { case (lbl, _, prob, _) =>
      new AveragedValue(count = 1L, value = -math.log(prob.toArray(lbl.toInt)))
    }.reduce(_ + _)
    avg.value
  }

  def binaryLogLoss: OpBinaryClassificationEvaluatorBase[SingleMetric] = Evaluators.BinaryClassification.custom(
    metricName = "BinarylogLoss",
    isLargerBetter = false,
    evaluateFn = logLossFun
  )

  def mulitLogLoss: OpMultiClassificationEvaluatorBase[SingleMetric] = Evaluators.MultiClassification.custom(
    metricName = "MultiClasslogLoss",
    isLargerBetter = false,
    evaluateFn = logLossFun
  )
}
