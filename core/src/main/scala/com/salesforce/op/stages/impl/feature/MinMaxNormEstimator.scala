package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{FeatureTypeFactory, Real}
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
  * Scaling estimator that rescales a numerical feature to have a range from 0 to 1
  *
  * @param uid         uid for instance
  * @param tti         type tag for input
  * @param tto         type tag for output
  * @param ttov        type tag for output value
  * @tparam I input feature type
  * @tparam O output feature type
  */
class MinMaxNormEstimator[I <: Real, O <: Real]
(
  uid: String = UID[MinMaxNormEstimator[_, _]]
)(implicit tti: TypeTag[I], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends UnaryEstimator[I, O](operationName = "minMaxNorm", uid = uid) {

  def fitFn(dataset: Dataset[O#Value]): UnaryModel[I, O] = {
    val grouped = dataset.groupBy()
    val maxVal = grouped.max().first().getDouble(0)
    val minVal = grouped.min().first().getDouble(0)

    val scalingArgs = LinearScalerArgs(1 / (maxVal - minVal), - minVal / (maxVal - minVal))
    val meta = ScalerMetadata(ScalingType.Linear, scalingArgs).toMetadata()
    setMetadata(meta)

    new MinMaxNormEstimatorModel(
      min = minVal,
      max = maxVal,
      seq = Seq(minVal, maxVal),
      map = Map("a" -> Map("b" -> 1.0, "c" -> 2.0), "d" -> Map.empty),
      operationName = operationName,
      uid = uid
    )
  }
}

final class MinMaxNormEstimatorModel[I <: Real, O <: Real]
(
  val min: Double,
  val max: Double,
  val seq: Seq[Double],
  val map: Map[String, Map[String, Double]],
  operationName: String, uid: String
)(implicit tti: TypeTag[I], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends UnaryModel[I, O](operationName = operationName, uid = uid) {

  private val ftFactory = FeatureTypeFactory[O]()

  def transformFn: I => O = r => {
    val scaled = r.v.map(v => (v - min) / (max - min))
    ftFactory.newInstance(scaled)
  }
}

