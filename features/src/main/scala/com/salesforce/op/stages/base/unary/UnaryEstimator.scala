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

package com.salesforce.op.stages.base.unary

import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.stages.OpPipelineStage1
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{Dataset, Encoder}
import org.apache.spark.util.ClosureUtils

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try

/**
 * Takes a single input feature and performs a fit operation in order to define a transformation (model)
 * for that feature.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttiv          type tag for input value
 * @param ttov          type tag for output value
 * @tparam I input feature type
 * @tparam O output feature type
 */
abstract class UnaryEstimator[I <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttiv: TypeTag[I#Value],
  val ttov: TypeTag[O#Value]
) extends Estimator[UnaryModel[I, O]] with OpPipelineStage1[I, O] {

  // Encoders & converters
  implicit val iEncoder: Encoder[I#Value] = FeatureSparkTypes.featureTypeEncoder[I]
  val iConvert = FeatureTypeSparkConverter[I]()

  /**
   * Function that fits the unary model
   */
  def fitFn(dataset: Dataset[I#Value]): UnaryModel[I, O]

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
  override def fit(dataset: Dataset[_]): UnaryModel[I, O] = {
    setInputSchema(dataset.schema).transformSchema(dataset.schema)

    val df = dataset.select(in1.name)
    val ds = df.map(r => iConvert.fromSpark(r.get(0)).value)
    val model = fitFn(ds)

    model
      .setParent(this)
      .setInput(in1.asFeatureLike[I])
      .setMetadata(getMetadata())
      .setOutputFeatureName(getOutputFeatureName)
  }

}

/**
 * Extend this class and return it from your [[UnaryEstimator]] fit function.
 * Takes a single input feature and produces a single new output feature using
 * the specified function. Performs row wise transformation specified in transformFn.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I input type
 * @tparam O output type
 */
abstract class UnaryModel[I <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[UnaryModel[I, O]] with OpTransformer1[I, O]
