/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
