/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.binary

import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.stages.OpPipelineStage2
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{Dataset, Encoder, Encoders}
import org.apache.spark.util.ClosureUtils

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try


/**
 * Takes a two input features and performs a fit operation in order to define a transformation for those
 * (or similar) features. This abstract class should be extended when settable parameters are needed within the fit
 * function
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tto           type tag for output
 * @param ttiv1         type tag for first input value
 * @param ttiv2         type tag for second input value
 * @param ttov          type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam O  output feature type
 */
abstract class BinaryEstimator[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tto: TypeTag[O],
  val ttiv1: TypeTag[I1#Value],
  val ttiv2: TypeTag[I2#Value],
  val ttov: TypeTag[O#Value]
) extends Estimator[BinaryModel[I1, I2, O]] with OpPipelineStage2[I1, I2, O] {

  // Encoders & converters
  implicit val i1Encoder: Encoder[I1#Value] = FeatureSparkTypes.featureTypeEncoder[I1]
  implicit val i2Encoder: Encoder[I2#Value] = FeatureSparkTypes.featureTypeEncoder[I2]
  implicit val tupleEncoder = Encoders.tuple[I1#Value, I2#Value](i1Encoder, i2Encoder)
  val convertI1 = FeatureTypeSparkConverter[I1]()
  val convertI2 = FeatureTypeSparkConverter[I2]()

  /**
   * Function that fits the binary model
   */
  def fitFn(dataset: Dataset[(I1#Value, I2#Value)]): BinaryModel[I1, I2, O]

  /**
   * Check if the stage is serializable
   *
   * @return Failure if not serializable
   */
  final override def checkSerializable: Try[Unit] = ClosureUtils.checkSerializable(fitFn _)

  /**
   * Spark operation on dataset to produce RDD for constructor fit function and then turn output function into a Model
   *
   * @param dataset input data for this stage
   * @return a fitted model that will perform the transformation specified by the function defined in constructor fit
   */
  override def fit(dataset: Dataset[_]): BinaryModel[I1, I2, O] = {
    setInputSchema(dataset.schema).transformSchema(dataset.schema)

    val df = dataset.select(in1.name, in2.name)
    val ds = df.map(r =>
      (convertI1.fromSpark(r.get(0)).value,
        convertI2.fromSpark(r.get(1)).value)
    )
    val model = fitFn(ds)

    model
      .setParent(this)
      .setInput(in1.asFeatureLike[I1], in2.asFeatureLike[I2])
      .setMetadata(getMetadata())
  }

}

/**
 * Extend this class and return it from your [[BinaryEstimator]] fit function.
 * Takes two input features and produces a single new output feature using
 * the specified function. Performs row wise transformation specified in transformFn.
 *
 * @param operationName unique name of the operation this stage performs
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 first input type
 * @tparam I2 second input type
 * @tparam O  output type
 */
abstract class BinaryModel[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[BinaryModel[I1, I2, O]] with OpTransformer2[I1, I2, O]
