/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.base.sequence

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.{OpPipelineStageN, OpTransformer}
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.util.ClosureUtils

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try

/**
 * Base trait for sequence transformers and models which take a sequence of input features of the same type and perform
 * the specified function on them to give a new output feature
 *
 * @tparam I input feature type
 * @tparam O output feature type
 */
trait OpTransformerN[I <: FeatureType, O <: FeatureType]
  extends Transformer with OpPipelineStageN[I, O] with OpTransformer {

  implicit val tti: TypeTag[I]

  /**
   * Function used to convert input to output
   */
  def transformFn: Seq[I] => O

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
    assert(inN.nonEmpty, "Inputs cannot be empty")
    val newSchema = setInputSchema(dataset.schema).transformSchema(dataset.schema)
    val functionUDF = FeatureSparkTypes.udfN[I, O](transformFn)
    val meta = newSchema(getOutputFeatureName).metadata
    val columns = inN.map(in => dataset.col(in.name))
    dataset.select(col("*"), functionUDF(struct(columns: _*)).as(getOutputFeatureName, meta))
  }

  private val transformNFn = FeatureSparkTypes.transformN[I, O](transformFn)
  override def transformRow: Row => Any = {
    val inNames = inN.map(_.name)
    (row: Row) => transformNFn(inNames.map(name => row.getAny(name)))
  }

}

/**
 * Transformer that takes a sequence input features of the same type and produces a single new output feature using
 * the specified function. Performs row wise transformation specified in transformFn.  This abstract class should
 * be extended when settable parameters are needed within the transform function.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I input feature type
 * @tparam O output feature type
 */
abstract class SequenceTransformer[I <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends OpTransformerN[I, O]


/**
 * Transformer that takes a sequence input features of the same type and produces a single new output feature using
 * the specified function. Performs row wise transformation specified in transformFn. This class should be extended
 * when no parameters are needed within the transform function.
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
final class SequenceLambdaTransformer[I <: FeatureType, O <: FeatureType]
(
  operationName: String,
  val transformFn: Seq[I] => O,
  uid: String = UID[SequenceLambdaTransformer[I, O]]
)(
  implicit tti: TypeTag[I],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends SequenceTransformer[I, O](operationName = operationName, uid = uid)
