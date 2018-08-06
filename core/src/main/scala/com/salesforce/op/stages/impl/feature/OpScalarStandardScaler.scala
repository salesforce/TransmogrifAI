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
import org.apache.spark.ml.feature.{StandardScaler => MLStandardScaler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.feature.{StandardScalerModel, StandardScaler => MLLibStandardScaler}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset


/**
 * Wraps Spark's native StandardScaler, which operates on vectors, to enable it to
 * operate directly on scalars.
 *
 */
class OpScalarStandardScaler
(
  private val estimator: MLStandardScaler = new MLStandardScaler().setWithMean(true).setWithStd(true),
  uid: String = UID[OpScalarStandardScaler]
) extends UnaryEstimator[RealNN, RealNN](
  operationName = "stdScaled",
  uid = uid
) {

  def fitFn(input: Dataset[Option[Double]]): UnaryModel[RealNN, RealNN] = {
    val vecData: RDD[OldVector] = input.rdd.map(v => OldVectors.fromML(Vectors.dense(v.get)))
    val internalScaler = new MLLibStandardScaler(withMean = estimator.getWithMean, withStd = estimator.getWithStd)
    val scalerModel = internalScaler.fit(vecData)

    new OpScalarStandardScalerModel(
      std = scalerModel.std.toArray,
      mean = scalerModel.mean.toArray,
      withStd = scalerModel.withStd,
      withMean = scalerModel.withMean,
      operationName = operationName,
      uid = uid
    )
  }

  def setWithMean(value: Boolean): this.type = {
    estimator.setWithMean(value)
    this
  }

  def setWithStd(value: Boolean): this.type = {
    estimator.setWithStd(value)
    this
  }

}


final class OpScalarStandardScalerModel private[op]
(
  val std: Array[Double],
  val mean: Array[Double],
  val withStd: Boolean,
  val withMean: Boolean,
  operationName: String,
  uid: String
) extends UnaryModel[RealNN, RealNN](operationName = operationName, uid = uid) {

  private val model = new StandardScalerModel(
    std = OldVectors.dense(std),
    mean = OldVectors.dense(mean),
    withStd = withStd,
    withMean = withMean
  )

  def transformFn: RealNN => RealNN = inScalar => {
    val inVector = Vectors.dense(inScalar.v.get)
    val outVector = model.transform(OldVectors.fromML(inVector))
    new RealNN(outVector(0))
  }

}
