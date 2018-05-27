/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.sparkwrappers.generic.SwThreeStageBinaryEstimator
import org.apache.spark.ml.SparkMLSharedParamConstants
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap

/**
 * Wraps a spark ML probabilistic classifier.  In SparkML, a probabilistic classifier is anything that inherits
 * from [[ProbabilisticClassifier]].  Examples of these probabilistic classifiers
 * include: RandomForestClassifier, NaiveBayes, LogisticRegression, and DecisionTreeClassifier.
 * These classifiers in spark ML output not a single column, but 3: (1) the raw unnormalized scores for each class,
 * (2) the probabilistic classification (normalized raw scores), and
 * (3) the labels of the output (e.g. max unnormalized score).
 * The defining characteristic of classifiers intended to be wrapped by this class is that they output a model which
 * takes in 2 columns as input (label and features) and output 3 columns as result.
 *
 * @param probClassifier the probabilistic classifier to wrap
 * @param uid            stage uid
 * @tparam E spark estimator to wrap
 * @tparam M spark model type returned by spark estimator wrapped
 */
class OpProbabilisticClassifierWrapper[E <: ProbabilisticClassifier[Vector, E, M],
M <: ProbabilisticClassificationModel[Vector, M]]
(
  val probClassifier: E,
  uid: String = UID[OpProbabilisticClassifierWrapper[E, M]]
) extends SwThreeStageBinaryEstimator[RealNN, OPVector, RealNN, OPVector, OPVector, M, E](
  inputParam1Name = SparkMLSharedParamConstants.LabelColName,
  inputParam2Name = SparkMLSharedParamConstants.FeaturesColName,
  outputParam1Name = SparkMLSharedParamConstants.PredictionColName,
  outputParam2Name = SparkMLSharedParamConstants.RawPredictionColName,
  outputParam3Name = SparkMLSharedParamConstants.ProbabilityColName,
  stage1OperationName = probClassifier.getClass.getSimpleName + "_" + SparkMLSharedParamConstants.PredictionColName ,
  stage2OperationName = probClassifier.getClass.getSimpleName + "_" + SparkMLSharedParamConstants.RawPredictionColName,
  stage3OperationName = probClassifier.getClass.getSimpleName + "_" + SparkMLSharedParamConstants.ProbabilityColName,
  // cloning below to prevent parameter changes to the underlying classifier outside the wrapper
  sparkMlStageIn = Option(probClassifier).map(_.copy(ParamMap.empty)),
  uid = uid
) {
  final protected def getSparkStage: E = getSparkMlStage().get
}
