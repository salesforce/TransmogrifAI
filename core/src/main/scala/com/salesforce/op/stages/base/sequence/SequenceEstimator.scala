/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.sequence

import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.stages.OpPipelineStageN
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Encoder, Encoders}
import org.apache.spark.util.ClosureUtils

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try

/**
 * Takes a sequence of input features of the same type and performs a fit operation in order to define a transformation
 * for those (or similar) features. This abstract class should be extended when settable parameters are needed within
 * the fit function
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for input
 * @param ttiv          type tag for input value
 * @param ttov          type tag for output value
 * @tparam I input feature type
 * @tparam O output feature type
 */
abstract class SequenceEstimator[I <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttiv: TypeTag[I#Value],
  val ttov: TypeTag[O#Value]
) extends Estimator[SequenceModel[I, O]] with OpPipelineStageN[I, O] {

  // Encoders & converters
  implicit val seqIEncoder: Encoder[Seq[I#Value]] = Encoders.kryo[Seq[I#Value]]
  val seqIConvert = FeatureTypeSparkConverter[I]()

  /**
   * Function that fits the sequence model
   */
  def fitFn(dataset: Dataset[Seq[I#Value]]): SequenceModel[I, O]

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
  override def fit(dataset: Dataset[_]): SequenceModel[I, O] = {
    assert(inN.nonEmpty, "Inputs cannot be empty")
    setInputSchema(dataset.schema).transformSchema(dataset.schema)

    val columns = inN.map(feature => col(feature.name))

    val df = dataset.select(columns: _*)
    val ds = df.map(_.toSeq.map(seqIConvert.fromSpark(_).value))
    val model = fitFn(ds)

    model
      .setParent(this)
      .setInput(inN.map(_.asFeatureLike[I]))
      .setMetadata(getMetadata())
  }

}


/**
 * Extend this class and return it from your [[SequenceEstimator]] fit function.
 * Takes a sequence of input features of the same type and produces a single
 * new output feature using the specified function. Performs row wise transformation specified in transformFn.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I input type
 * @tparam O output type
 */
abstract class SequenceModel[I <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[SequenceModel[I, O]] with OpTransformerN[I, O]
