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
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import scala.reflect.runtime.universe.TypeTag
import org.apache.spark.sql.functions.stddev_samp
import org.apache.spark.sql.Dataset

/**
  * Scaling estimator that rescales a numerical feature to have min = 0 and variance = 1
  *
  * @param uid         uid for instance
  * @param tti         type tag for input
  * @param tto         type tag for output
  * @param ttov        type tag for output value
  * @tparam I input feature type
  * @tparam O output feature type
  */
class StandardMinEstimator[I <: Real, O <: Real]
(
  uid: String = UID[StandardMinEstimator[_, _]]
)(implicit tti: TypeTag[I], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends UnaryEstimator[I, O](operationName = "standardMin", uid = uid) {

  def fitFn(dataset: Dataset[O#Value]): UnaryModel[I, O] = {
    val grouped = dataset.groupBy()
    val minVal = grouped.min().first().getDouble(0)

    val stdVal = dataset.agg(stddev_samp(dataset.columns.head)).first().getDouble(0)

    val scalingArgs = LinearScalerArgs(1 / stdVal, - minVal / stdVal)
    val meta = ScalerMetadata(ScalingType.Linear, scalingArgs).toMetadata()
    setMetadata(meta)


    new StandardMinEstimatorModel(
      min = minVal,
      std = stdVal,
      operationName = operationName,
      uid = uid
    )
  }
}

final class StandardMinEstimatorModel[I <: Real, O <: Real]
(
  val min: Double,
  val std: Double,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[I], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends UnaryModel[I, O](operationName = operationName, uid = uid) {

  private val ftFactory = FeatureTypeFactory[O]()

  def transformFn: I => O = r => {
    val scaled = r.v.map(v => (v - min) / std)
    ftFactory.newInstance(scaled)
  }
}
