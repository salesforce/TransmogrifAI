/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.ternary

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.{OpPipelineStage3, OpTransformer}
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.util.ClosureUtils

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try

/**
 * Base trait for ternary transformers and models which take three input features and perform specified function on
 * them to give a new output feature
 *
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam O  output feature type
 */
trait OpTransformer3[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, O <: FeatureType]
  extends Transformer with OpPipelineStage3[I1, I2, I3, O] with OpTransformer {

  implicit val tti1: TypeTag[I1]
  implicit val tti2: TypeTag[I2]
  implicit val tti3: TypeTag[I3]

  /**
   * Function used to convert input to output
   */
  def transformFn: (I1, I2, I3) => O

  /**
   * Check if the stage is serializable
   *
   * @return Failure if not serializable
   */
  final override def checkSerializable: Try[Unit] = ClosureUtils.checkSerializable(transformFn)

  /**
   * Spark operation on dataset to produce new output feature column using defined function
   *
   * @param dataset input data for this stage
   * @return a new dataset containing a column for the transformed feature
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val newSchema = setInputSchema(dataset.schema).transformSchema(dataset.schema)
    val functionUDF = FeatureSparkTypes.udf3[I1, I2, I3, O](transformFn)
    val meta = newSchema(outputName).metadata
    dataset.select(col("*"), functionUDF(col(in1.name), col(in2.name), col(in3.name)).as(outputName, meta))
  }

  private val transform3Fn = FeatureSparkTypes.transform3[I1, I2, I3, O](transformFn)
  override def transformRow: Row => Any = {
    val (in1name, in2name, in3name) = (in1.name, in2.name, in3.name)
    (row: Row) => transform3Fn(row.getAny(in1name), row.getAny(in2name), row.getAny(in3name))
  }

}


/**
 * Transformer that takes three input features and produces a single new output feature using the specified function.
 * Performs row wise transformation specified in transformFn. This abstract class should be extended when settable
 * parameters are needed within the transform function.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tti3          type tag for third input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam O  output feature type
 */
abstract class TernaryTransformer[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tti3: TypeTag[I3],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends OpTransformer3[I1, I2, I3, O]


/**
 * Transformer that takes three input features and produces a single new output feature using the specified function.
 * Performs row wise transformation specified in transformFn. This class should be extended when no
 * parameters are needed within the transform function.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param transformFn   function used to convert input to output
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tti3          type tag for third input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam O  output feature type
 */
final class TernaryLambdaTransformer[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, O <: FeatureType]
(
  operationName: String,
  val transformFn: (I1, I2, I3) => O,
  uid: String = UID[TernaryLambdaTransformer[I1, I2, I3, O]]
)(
  implicit tti1: TypeTag[I1],
  tti2: TypeTag[I2],
  tti3: TypeTag[I3],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends TernaryTransformer[I1, I2, I3, O](operationName = operationName, uid = uid)
