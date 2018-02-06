/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen.templates
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types.{OPVector, RealNN}
import com.salesforce.op.stages.impl.regression.RegressionModelSelector

/**
 * This is a template for generating some code
 */
trait RegressionTemplate {
  val label: Feature[RealNN]
  val checkedFeatures: Feature[OPVector]
  // BEGIN
  val pred = RegressionModelSelector()
    .setInput(label, checkedFeatures)
    .getOutput()

  val evaluator =
    Evaluators.Regression()
      .setLabelCol(label).setPredictionCol(pred)
  // END
}
