/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen.templates

import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types.{OPVector, RealNN}
import com.salesforce.op.stages.impl.classification.MultiClassificationModelSelector

/**
 * This is a template for generating some code
 */
trait MultiClassificationTemplate {
  val label: Feature[RealNN]
  val checkedFeatures: Feature[OPVector]
  // BEGIN
  val (pred, raw, prob) = MultiClassificationModelSelector()
    .setInput(label, checkedFeatures)
    .getOutput()

  val evaluator =
    Evaluators.MultiClassification()
      .setLabelCol(label).setPredictionCol(pred).setRawPredictionCol(raw)
  // END
}
