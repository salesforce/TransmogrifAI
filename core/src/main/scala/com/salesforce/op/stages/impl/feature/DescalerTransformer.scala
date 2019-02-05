package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{FeatureTypeFactory, Prediction, Real}
import com.salesforce.op.stages.base.binary.BinaryTransformer

import scala.reflect.runtime.universe.TypeTag
import scala.util.{Failure, Success}

/**
 *
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam O  output feature type
 */
final class DescalerTransformer[I1 <: Real, I2 <: Real, O <: Real]
(
  uid: String = UID[DescalerTransformer[_, _, _]]
)(implicit tti1: TypeTag[I1], tti2: TypeTag[I2], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends BinaryTransformer[I1, I2, O](operationName = "descaler", uid = uid) {

  private val ftFactory = FeatureTypeFactory[O]()

  @transient private lazy val meta = getInputSchema()(in2.name).metadata
  @transient private lazy val scalerMeta: ScalerMetadata = ScalerMetadata(meta) match {
    case Success(sm) => sm
    case Failure(error) =>
      throw new RuntimeException(s"Failed to extract scaler metadata for input feature '${in2.name}'", error)
  }
  @transient private lazy val scaler = Scaler(scalerMeta.scalingType, scalerMeta.scalingArgs)

  def transformFn: (I1, I2) => O = (v, _) => {
    val descaled = v.toDouble.map(scaler.descale)
    ftFactory.newInstance(descaled)
  }

}

/**
 *  - 1st input feature is the prediction feature to descale
 *  - 2nd input feature is the scaled label to get the metadata from
 */
final class PredictionDescaler[I <: Real, O <: Real]
(
  uid: String = UID[DescalerTransformer[_, _, _]]
)(implicit tti2: TypeTag[I], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends BinaryTransformer[Prediction, I, O](operationName = "descaler", uid = uid) {

  private val ftFactory = FeatureTypeFactory[O]()

  @transient private lazy val meta = getInputSchema()(in2.name).metadata
  @transient private lazy val scalerMeta: ScalerMetadata = ScalerMetadata(meta) match {
    case Success(sm) => sm
    case Failure(error) =>
      throw new RuntimeException(s"Failed to extract scaler metadata for input feature '${in2.name}'", error)
  }
  @transient private lazy val scaler = Scaler(scalerMeta.scalingType, scalerMeta.scalingArgs)

  def transformFn: (Prediction, I) => O = (v, _) => {
    val descaled = scaler.descale(v.prediction)
    ftFactory.newInstance(descaled)
  }
}
