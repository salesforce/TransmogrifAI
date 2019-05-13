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

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.sparkwrappers.generic.SwUnaryTransformer
import org.apache.spark.ml.SparkMLSharedParamConstants
import org.apache.spark.ml.SparkMLSharedParamConstants.InOutTransformer
import org.apache.spark.ml.param.ParamMap

import scala.reflect.runtime.universe.TypeTag

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
