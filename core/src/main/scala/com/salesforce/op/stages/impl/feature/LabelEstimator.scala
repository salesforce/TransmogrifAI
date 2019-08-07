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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{FeatureTypeFactory, Real}
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.stages.impl.feature.LabelScaler._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.stddev_samp

import scala.reflect.runtime.universe.TypeTag

/**
 * Scaling estimator that rescales a numerical feature to have a range from 0 to 1
 *
 * @param labelScalingType   type of label scaling
 * @param uid                uid for instance
 * @param tti                type tag for input
 * @param tto                type tag for output
 * @param ttov               type tag for output value
 * @tparam I input feature type
 * @tparam O output feature type
 */
class LabelEstimator[I <: Real, O <: Real]
(
  labelScalingType: LabelScaler,
  uid: String = UID[LabelEstimator[_, _]]
)(implicit tti: TypeTag[I], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends UnaryEstimator[I, O](operationName = "labelEstimator", uid = uid) {

  def fitFn(dataset: Dataset[O#Value]): UnaryModel[I, O] = {
    val meanVal = dataset.groupBy().mean().head().getDouble(0)
    val minVal = dataset.groupBy().min().head().getDouble(0)
    val maxVal = dataset.groupBy().max().head().getDouble(0)
    val stdVal = dataset.agg(stddev_samp(dataset.columns.head)).head().getDouble(0)

    val (slope, intercept) = labelScalingType match {
      case Norm => (1 / stdVal, - meanVal / stdVal)
      case MinMax => (1 / (maxVal - minVal), - minVal / (maxVal - minVal))
      case StandardZeroMin => (1 / stdVal, - minVal / stdVal)
      case _ => throw new NotImplementedError(f"$labelScalingType has not yet been implemented")
    }

    val scalingArgs = LinearScalerArgs(slope, intercept)
    val meta = ScalerMetadata(ScalingType.Linear, scalingArgs).toMetadata()
    setMetadata(meta)

    new LabelEstimatorModel[I, O](
      slope = slope,
      intercept = intercept,
      operationName = operationName,
      uid = uid
    )

  }
}

final class LabelEstimatorModel[I <: Real, O <: Real]
(
  val slope: Double,
  val intercept: Double,
  operationName: String, uid: String
)(implicit tti: TypeTag[I], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends UnaryModel[I, O](operationName = operationName, uid = uid) {

  private val ftFactory = FeatureTypeFactory[O]()

  def transformFn: I => O = r => {
    val scaled = r.v.map(v => v * slope + intercept)
    ftFactory.newInstance(scaled)
  }
}
