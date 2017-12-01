/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.quaternary

import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.stages.OpPipelineStage4
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{Dataset, Encoder, Encoders}
import org.apache.spark.util.ClosureUtils

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try

/**
 * Takes a four input features and performs a fit operation in order to define a transformation for those
 * (or similar) features. This abstract class should be extended when settable parameters are needed within the fit
 * function
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tti3          type tag for third input
 * @param tti4          type tag for fourth input
 * @param tto           type tag for input
 * @param ttiv1         type tag for first input value
 * @param ttiv2         type tag for second input value
 * @param ttiv3         type tag for third input value
 * @param ttiv4         type tag for fourth input value
 * @param ttov          type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam I4 fourth input feature type
 * @tparam O  output feature type
 */
abstract class QuaternaryEstimator[I1 <: FeatureType,
I2 <: FeatureType, I3 <: FeatureType, I4 <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tti3: TypeTag[I3],
  val tti4: TypeTag[I4],
  val tto: TypeTag[O],
  val ttiv1: TypeTag[I1#Value],
  val ttiv2: TypeTag[I2#Value],
  val ttiv3: TypeTag[I3#Value],
  val ttiv4: TypeTag[I4#Value],
  val ttov: TypeTag[O#Value]
) extends Estimator[QuaternaryModel[I1, I2, I3, I4, O]] with OpPipelineStage4[I1, I2, I3, I4, O] {

  // Encoders & converters
  implicit val i1Encoder: Encoder[I1#Value] = FeatureSparkTypes.featureTypeEncoder[I1]
  implicit val i2Encoder: Encoder[I2#Value] = FeatureSparkTypes.featureTypeEncoder[I2]
  implicit val i3Encoder: Encoder[I3#Value] = FeatureSparkTypes.featureTypeEncoder[I3]
  implicit val i4Encoder: Encoder[I4#Value] = FeatureSparkTypes.featureTypeEncoder[I4]
  implicit val tupleEncoder =
    Encoders.tuple[I1#Value, I2#Value, I3#Value, I4#Value](i1Encoder, i2Encoder, i3Encoder, i4Encoder)
  val convertI1 = FeatureTypeSparkConverter[I1]()
  val convertI2 = FeatureTypeSparkConverter[I2]()
  val convertI3 = FeatureTypeSparkConverter[I3]()
  val convertI4 = FeatureTypeSparkConverter[I4]()

  /**
   * Function that fits the quaternary model
   */
  def fitFn(dataset: Dataset[(I1#Value, I2#Value, I3#Value, I4#Value)]): QuaternaryModel[I1, I2, I3, I4, O]

  /**
   * Check if the stage is serializable
   *
   * @return Failure if not serializable
   */
  final override def checkSerializable: Try[Unit] = ClosureUtils.checkSerializable(fitFn _)

  /**
   * Spark operation on dataset to produce Dataset
   * for constructor fit function and then turn output function into a Model
   *
   * @param dataset input data for this stage
   * @return a fitted model that will perform the transformation specified by the function defined in constructor fit
   */
  override def fit(dataset: Dataset[_]): QuaternaryModel[I1, I2, I3, I4, O] = {
    transformSchema(dataset.schema)
    setInputSchema(dataset.schema)

    val df = dataset.select(in1.name, in2.name, in3.name, in4.name)
    val ds = df.map(r =>
      (convertI1.fromSpark(r.get(0)).value,
        convertI2.fromSpark(r.get(1)).value,
        convertI3.fromSpark(r.get(2)).value,
        convertI4.fromSpark(r.get(3)).value)
    )
    val model = fitFn(ds)

    model
      .setParent(this)
      .setInput(in1.asFeatureLike[I1], in2.asFeatureLike[I2], in3.asFeatureLike[I3], in4.asFeatureLike[I4])
      .setMetadata(getMetadata())
  }
}


/**
 * Extend this class and return it from your [[QuaternaryEstimator]] fit function.
 * Takes four input features and produces a single new output feature using
 * the specified function. Performs row wise transformation specified in transformFn.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tti3          type tag for third input
 * @param tti4          type tag for fourth input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 first input type
 * @tparam I2 second input type
 * @tparam I3 third input type
 * @tparam I4 fourth input type
 * @tparam O  output type
 */
abstract class QuaternaryModel[I1 <: FeatureType,
I2 <: FeatureType, I3 <: FeatureType, I4 <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tti3: TypeTag[I3],
  val tti4: TypeTag[I4],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[QuaternaryModel[I1, I2, I3, I4, O]] with OpTransformer4[I1, I2, I3, I4, O]

