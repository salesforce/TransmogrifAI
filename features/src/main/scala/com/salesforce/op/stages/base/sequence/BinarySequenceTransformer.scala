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

package com.salesforce.op.stages.base.sequence

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.{OpPipelineStage2N, OpTransformer}
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.util.ClosureUtils
import org.apache.spark.sql.functions._

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try

/**
 * Base trait for binary sequence transformers and models which take a single feature as first argument and a
 * sequence of input features of a similar type and perform the specified function on them to give a
 * new output feature
 *
 * @tparam I1 input feature of singular type
 * @tparam I2 input feature of sequence type
 * @tparam O output feature type
 */
trait OpTransformer2N[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType]
  extends Transformer with OpPipelineStage2N[I1, I2, O] with OpTransformer {

  implicit val tti1: TypeTag[I1]
  implicit val tti2: TypeTag[I2]

  /**
   * Function used to convert input to output
   */
  def transformFn: (I1, Seq[I2]) => O

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
    assert(getTransientFeatures.size > 1, "Inputs cannot be empty")
    val newSchema = setInputSchema(dataset.schema).transformSchema(dataset.schema)
    val functionUDF = FeatureSparkTypes.udf2N[I1, I2, O](transformFn)
    val meta = newSchema(getOutputFeatureName).metadata
    val columns = getTransientFeatures().map(in => dataset.col(in.name))
    dataset.select(col("*"), functionUDF(struct(columns: _*)).as(getOutputFeatureName, meta))
  }

  private lazy val transformNFn = FeatureSparkTypes.transform2N[I1, I2, O](transformFn)
  override def transformKeyValue: KeyValue => Any = {
    val inName1 = in1.name
    val inNames = inN.map(_.name)
    (kv: KeyValue) => transformNFn(kv(inName1), inNames.map(name => kv(name)))
  }

}

/**
 * Transformer that takes a single feature of type I1 and a sequence of features o type I2 and produces
 * a single new output feature using the specified function. Performs row wise transformation specified
 * in transformFn.  This abstract class should be extended when settable parameters are needed within
 * the transform function.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for single input
 * @param tti2          type tag for sequence input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 input single feature type
 * @tparam I2 input sequence feature type
 * @tparam O output feature type
 */
abstract class BinarySequenceTransformer[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends OpTransformer2N[I1, I2, O]

/**
 * Transformer that takes a single feature of type I1 and a sequence of features of type I2 and produces
 * a single new output feature using the specified function. Performs row wise transformation specified
 * in transformFn. This class should be extended when no parameters are needed within the transform function.
 *
 * @param operationName unique name of the operation this stage performs
 * @param transformFn   function used to convert input to output
 * @param uid           uid for instance
 * @param tti1          type tag for single input
 * @param tti2          type tag for sequence input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 input single feature type
 * @tparam I2 input sequence feature type
 * @tparam O output feature type
 */
final class BinarySequenceLambdaTransformer[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType]
(
  operationName: String,
  val transformFn: (I1, Seq[I2]) => O,
  uid: String = UID[BinarySequenceLambdaTransformer[I1, I2, O]]
)(
  implicit tti1: TypeTag[I1],
  tti2: TypeTag[I2],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends BinarySequenceTransformer[I1, I2, O](operationName = operationName, uid = uid)
