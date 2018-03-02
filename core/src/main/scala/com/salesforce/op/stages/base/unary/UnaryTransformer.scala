/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.unary

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.{OpPipelineStage1, OpTransformer}
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.util.ClosureUtils

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try

/**
 * Base trait for unary transformers and models which take one input feature and perform specified function on it to
 * give a new output feature
 *
 * @tparam I input feature type
 * @tparam O output feature type
 */
trait OpTransformer1[I <: FeatureType, O <: FeatureType]
  extends Transformer with OpPipelineStage1[I, O] with OpTransformer {

  /**
   * Function used to convert input to output
   */
  def transformFn: I => O

  implicit val tti: TypeTag[I]

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
    val functionUDF = FeatureSparkTypes.udf1[I, O](transformFn)
    val meta = newSchema(outputName).metadata
    dataset.select(col("*"), functionUDF(col(in1.name)).as(outputName, meta))
  }

  private val transform1Fn = FeatureSparkTypes.transform1[I, O](transformFn)
  override def transformRow: Row => Any = {
    val inName = in1.name
    (r: Row) => transform1Fn(r.getAny(inName))
  }

}


/**
 * Transformer that takes a single input feature and produces a single new output feature using the specified function.
 * Performs row wise transformation specified in transformFn. This abstract class should be extended when settable
 * parameters are needed within the transform function.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I input feature type
 * @tparam O output feature type
 */
abstract class UnaryTransformer[I <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends OpTransformer1[I, O]


/**
 * Transformer that takes a single input feature and produces a single new output feature using the specified function.
 * Performs row wise transformation specified in transformFn.
 *
 * @param operationName unique name of the operation this stage performs
 * @param transformFn   function used to convert input to output
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I input feature type
 * @tparam O output feature type
 */
final class UnaryLambdaTransformer[I <: FeatureType, O <: FeatureType]
(
  operationName: String,
  val transformFn: I => O,
  uid: String = UID[UnaryLambdaTransformer[I, O]]
)(
  implicit tti: TypeTag[I],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends UnaryTransformer[I, O](operationName = operationName, uid = uid)
