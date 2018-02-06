/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
