/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml

import com.salesforce.op.UID
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasLabelCol, _}

/**
 * Class containing constants used by spark ML for common parameters (mostly column names).
 */
case object SparkMLSharedParamConstants
  extends HasLabelCol
    with HasFeaturesCol
    with HasPredictionCol
    with HasRawPredictionCol
    with HasProbabilityCol
    with HasInputCol
    with HasOutputCol {
  // These are the typical names for the parameters in the sparkML algorithms:
  // target variable
  val LabelColName = labelCol.name
  // feature vector
  val FeaturesColName = featuresCol.name
  // response variable
  val PredictionColName = predictionCol.name

  // these are the names for input/ouput for estimators/transformers that transform a single column to another
  // single columns
  val InputColName = inputCol.name
  val OutputColName = outputCol.name

  // These param names below are used in probabilistic classifiers in sparkML, where 'raw' simply
  // represents an unnormalized version of the other.
  val RawPredictionColName = rawPredictionCol.name
  val ProbabilityColName = probabilityCol.name

  // ********************************************************************************
  // just adding these below so I can have this object (static class) for constants
  override def copy(extra: ParamMap): SparkMLSharedParamConstants.type = copyValues(this, extra)
  override val uid: String = UID(this.getClass)
  // ********************************************************************************

  type InOutTransformer = Transformer with HasInputCol with HasOutputCol
}


