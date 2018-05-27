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

package org.apache.spark.ml

import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.base.binary.OpTransformer2
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._

/**
 * Allows conversion from spark models to models that follow the OP convention of having a
 * transformFn that can be called on a single row rather than the whole dataframe
 */
object SparkModelConverter {

  def toOP[T <: Transformer](
    model: Option[T],
    isMultinomial: Boolean = false
  ): OpTransformer2[RealNN, OPVector, Prediction] = {
    model match {
      case None => throw new RuntimeException("no model found")
      case Some(m: LogisticRegressionModel) =>
        new OpLogisticRegressionModel(m.coefficientMatrix, m.interceptVector, m.numClasses, isMultinomial)
      case Some(m: RandomForestClassificationModel) =>
        new OpRandomForestClassificationModel(m.trees, m.numFeatures, m.numClasses)
      case Some(m: NaiveBayesModel) =>
        new OpNaiveBayesModel(m.pi, m.theta, m.oldLabels, if (isMultinomial) "multinomial" else "bernoulli")
      case Some(m: DecisionTreeClassificationModel) =>
        new OpDecisionTreeClassificationModel(m.rootNode, m.numFeatures, m.numClasses)
      case Some(m: LinearRegressionModel) =>
        new OpLinearPredictionModel(m.coefficients, m.intercept)
      case Some(m: RandomForestRegressionModel) =>
        new OpRandomForestRegressionModel(m.trees, m.numFeatures)
      case Some(m: GBTRegressionModel) =>
        new OpGBTRegressionModel(m.trees, m.treeWeights, m.numFeatures)
      case Some(m: DecisionTreeRegressionModel) =>
        new OpDecisionTreeRegressionModel(m.rootNode, m.numFeatures)
      case m => throw new RuntimeException(s"model conversion not implemented for model $m")
    }
  }
}
