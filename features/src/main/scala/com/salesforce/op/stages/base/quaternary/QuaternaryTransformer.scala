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

package com.salesforce.op.stages.base.quaternary

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.{OpPipelineStage4, OpTransformer}
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.util.ClosureUtils

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try


/**
 * Base trait for quaternary transformers and models which take four input features and perform specified function on
 * them to give a new output feature
 *
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam I4 fourth input feature type
 * @tparam O  output feature type
 */
trait OpTransformer4[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, I4 <: FeatureType, O <: FeatureType]
  extends Transformer with OpPipelineStage4[I1, I2, I3, I4, O] with OpTransformer {

  implicit val tti1: TypeTag[I1]
  implicit val tti2: TypeTag[I2]
  implicit val tti3: TypeTag[I3]
  implicit val tti4: TypeTag[I4]

  /**
   * Function used to convert input to output
   */
  def transformFn: (I1, I2, I3, I4) => O

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
    val functionUDF = FeatureSparkTypes.udf4[I1, I2, I3, I4, O](transformFn)
    val meta = newSchema(getOutputFeatureName).metadata
    dataset.select(col("*"),
      functionUDF(col(in1.name), col(in2.name), col(in3.name), col(in4.name)).as(getOutputFeatureName, meta))
  }

  private val transform4Fn = FeatureSparkTypes.transform4[I1, I2, I3, I4, O](transformFn)
  override def transformKeyValue: KeyValue => Any = {
    val (in1name, in2name, in3name, in4name) = (in1.name, in2.name, in3.name, in4.name)
    (kv: KeyValue) => transform4Fn(kv(in1name), kv(in2name), kv(in3name), kv(in4name))
  }

}


/**
 * Transformer that takes four input features and produces a single new output feature using the specified function.
 * Performs row wise transformation specified in transformFn. This abstract class should be extended when settable
 * parameters are needed within the transform function.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tti3          type tag for third input
 * @param tti4          type tag for fourth input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam I4 fourth input feature type
 * @tparam O  output feature type
 */
abstract class QuaternaryTransformer[I1 <: FeatureType,
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
) extends OpTransformer4[I1, I2, I3, I4, O]


/**
 * Transformer that takes four input features and produces a single new output feature using the specified function.
 * Performs row wise transformation specified in transformFn. This class should be extended when no
 * parameters are needed within the transform function.
 *
 * @param operationName unique name of the operation this stage performs
 * @param transformFn   function used to convert input to output
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tti3          type tag for third input
 * @param tti4          type tag for fourth input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam I4 fourth input feature type
 * @tparam O  output feature type
 */
final class QuaternaryLambdaTransformer[I1 <: FeatureType,
I2 <: FeatureType, I3 <: FeatureType, I4 <: FeatureType, O <: FeatureType]
(
  operationName: String,
  val transformFn: (I1, I2, I3, I4) => O,
  uid: String = UID[QuaternaryLambdaTransformer[I1, I2, I3, I4, O]]
)(
  implicit tti1: TypeTag[I1],
  tti2: TypeTag[I2],
  tti3: TypeTag[I3],
  tti4: TypeTag[I4],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends QuaternaryTransformer[I1, I2, I3, I4, O](operationName = operationName, uid = uid)

