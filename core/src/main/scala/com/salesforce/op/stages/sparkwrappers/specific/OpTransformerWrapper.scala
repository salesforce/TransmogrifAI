/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.sparkwrappers.generic.SwUnaryTransformer
import org.apache.spark.ml.SparkMLSharedParamConstants
import org.apache.spark.ml.SparkMLSharedParamConstants.InOutTransformer
import org.apache.spark.ml.param.ParamMap

import scala.reflect.runtime.universe.TypeTag

// TODO: all the transformers that inherit traits HasInputCol and HasOutputCol should really extend
// org.apache.spark.ml.UnaryTransformer, so can add a PR to spark so we can then move this class to our namespace.

/**
 * Wraps a spark ML transformer with setable input and output columns.  Those transformers that fall in this case,
 * include those that inherit from org.apache.spark.ml.UnaryEstimator, as well as others such as OneHotEncoder,
 * [[org.apache.spark.ml.feature.Binarizer]], [[org.apache.spark.ml.feature.VectorSlicer]],
 * [[org.apache.spark.ml.feature.HashingTF]], [[org.apache.spark.ml.feature.StopWordsRemover]],
 * [[org.apache.spark.ml.feature.IndexToString]], [[org.apache.spark.ml.feature.StringIndexer]].
 * Their defining characteristic is that they take one column as input, and output one column as result.
 *
 * @param transformer The spark ML transformer that's being wrapped
 * @param uid         stage uid
 * @tparam I The type of the input feature
 * @tparam O The type of the output feature (result of transformation)
 * @tparam T type of spark transformer to wrap
 */
class OpTransformerWrapper[I <: FeatureType, O <: FeatureType, T <: InOutTransformer]
(
  val transformer: T,
  uid: String = UID[OpTransformerWrapper[I, O, T]]
)(
  implicit tti: TypeTag[I],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends SwUnaryTransformer[I, O, T](
  inputParamName = SparkMLSharedParamConstants.InputColName,
  outputParamName = SparkMLSharedParamConstants.OutputColName,
  operationName = transformer.getClass.getSimpleName,
  // cloning below to prevent parameter changes to the underlying transformer outside the wrapper
  sparkMlStageIn = Option(transformer).map(_.copy(ParamMap.empty).asInstanceOf[T]),
  uid = uid
)
