/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.stages.impl.classification.Impurity._
import com.salesforce.op.stages.impl.classification.ModelType.Multinomial
import com.salesforce.op.stages.impl.regression.LossType.Squared
import com.salesforce.op.stages.impl.regression.Solver.Auto

object DefaultSelectorParams {

  val MaxDepth = Seq(3, 6, 12) // for trees spark default 5
  val MaxBin = 32 // bins for cont variables in trees - 32 is spark default
  val MinInstancesPerNode = Seq(10, 100) // spark default 1
  val MinInfoGain = Seq(0.001, 0.01, 0.1) // spark default 0
  val Regularization = Seq(0.001, 0.01, 0.1) // spark default 0
  val MaxIterLin = 50 // spark default is 100
  val MaxIterTree = 20 // spark default is 20
  val SubsampleRate = 1.0 // sample of data used for tree fits spark default 1.0
  val StepSize = 0.1 // spark default 0.1
  val ImpurityReg = Variance // spark default variance
  val ImpurityClass = Gini // spark default gini
  val ElasticNet = 0.1 // turn on spark default 0
  val MaxTrees = 50 // spark default 20
  val Standardized = true // standardize for linear spark default true
  val Tol = 1E-6 // termination tol spark default 1E-6
  val TreeLossType = Squared // gbt tree loss measure spark default squared
  val RegSolver = Auto // regression solver spark default auto
  val FitIntercept = true // fit intercept spark default true
  val NbSmoothing = 1.0 // spark default 1.0
  val NbModel = Multinomial // spark default multinomial

}
