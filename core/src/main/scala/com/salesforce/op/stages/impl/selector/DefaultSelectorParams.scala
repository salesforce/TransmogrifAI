/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.stages.impl.classification.Impurity._

object DefaultSelectorParams {

  val MaxDepth = Array(5, 10, 20) // for trees spark default 5
  val MaxBin = Array(32) // bins for cont variables in trees - 32 is spark default
  val MinInstancesPerNode = Array(1, 10, 15, 20, 30) // spark default 1
  val MinInfoGain = Array(0, 0.001, 0.01) // spark default 0
  val Regularization = Array(0, 0.001, 0.01, 0.05, 0.1) // spark default 0
  val MaxIterLin = Array(10, 50, 100) // spark default is 100
  val MaxIterTree = Array(20) // spark default is 20
  val SubsampleRate = Array(1.0) // sample of data used for tree fits spark default 1.0
  val StepSize = Array(0.1) // spark default 0.1
  val ImpurityReg = Array(Variance.sparkName) // spark default variance
  val ImpurityClass = Array(Gini.sparkName) // spark default gini
  val ElasticNet = Array(0, 0.001, 0.01, 0.1, 0.5, 1) // turn on spark default 0
  val MaxTrees = Array(50) // spark default 20
  val Standardized = Array(true) // standardize for linear spark default true
  val Tol = Array(1E-6) // termination tol spark default 1E-6
  val TreeLossType = Array("squared") // gbt tree loss measure spark default squared
  val RegSolver = Array("auto") // regression solver spark default auto
  val FitIntercept = Array(true) // fit intercept spark default true
  val NbSmoothing = Array(1.0) // spark default 1.0
  val DistFamily = Array("gaussian", "poisson") // generalized linear model link family

}
