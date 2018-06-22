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

package com.salesforce.op.stages.base.ternary

import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.stages.OpPipelineStage3
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{Dataset, Encoder, Encoders}
import org.apache.spark.util.ClosureUtils

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try

/**
 * Takes a three input features and performs a fit operation in order to define a transformation for those
 * (or similar) features. This abstract class should be extended when settable parameters are needed within the fit
 * function
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tti3          type tag for third input
 * @param tto           type tag for input
 * @param ttiv1         type tag for first input value
 * @param ttiv2         type tag for second input value
 * @param ttiv3         type tag for third input value
 * @param ttov          type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam O  output feature type
 */
abstract class TernaryEstimator[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tti3: TypeTag[I3],
  val tto: TypeTag[O],
  val ttiv1: TypeTag[I1#Value],
  val ttiv2: TypeTag[I2#Value],
  val ttiv3: TypeTag[I3#Value],
  val ttov: TypeTag[O#Value]
) extends Estimator[TernaryModel[I1, I2, I3, O]] with OpPipelineStage3[I1, I2, I3, O] {

  // Encoders & converters
  implicit val i1Encoder: Encoder[I1#Value] = FeatureSparkTypes.featureTypeEncoder[I1]
  implicit val i2Encoder: Encoder[I2#Value] = FeatureSparkTypes.featureTypeEncoder[I2]
  implicit val i3Encoder: Encoder[I3#Value] = FeatureSparkTypes.featureTypeEncoder[I3]
  implicit val tupleEncoder = Encoders.tuple[I1#Value, I2#Value, I3#Value](i1Encoder, i2Encoder, i3Encoder)
  val convertI1 = FeatureTypeSparkConverter[I1]()
  val convertI2 = FeatureTypeSparkConverter[I2]()
  val convertI3 = FeatureTypeSparkConverter[I3]()

  /**
   * Function that fits the ternary model
   */
  def fitFn(dataset: Dataset[(I1#Value, I2#Value, I3#Value)]): TernaryModel[I1, I2, I3, O]

  /**
   * Check if the stage is serializable
   *
   * @return Failure if not serializable
   */
  final override def checkSerializable: Try[Unit] = ClosureUtils.checkSerializable(fitFn _)

  /**
   * Spark operation on dataset to produce Dataset[(I1, I2, I3)]
   * for constructor fit function and then turn output function into a Model
   *
   * @param dataset input data for this stage
   * @return a fitted model that will perform the transformation specified by the function defined in constructor fit
   */
  override def fit(dataset: Dataset[_]): TernaryModel[I1, I2, I3, O] = {
    setInputSchema(dataset.schema).transformSchema(dataset.schema)

    val df = dataset.select(in1.name, in2.name, in3.name)
    val ds = df.map(r =>
      (convertI1.fromSpark(r.get(0)).value,
        convertI2.fromSpark(r.get(1)).value,
        convertI3.fromSpark(r.get(2)).value)
    )
    val model = fitFn(ds)

    model
      .setParent(this)
      .setInput(in1.asFeatureLike[I1], in2.asFeatureLike[I2], in3.asFeatureLike[I3])
      .setMetadata(getMetadata())
      .setOutputFeatureName(getOutputFeatureName)
  }

}

/**
 * Extend this class and return it from your [[TernaryEstimator]] fit function.
 * Takes three input features and produces a single new output feature using
 * the specified function. Performs row wise transformation specified in transformFn.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @param tti3          type tag for third input
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 first input type
 * @tparam I2 second input type
 * @tparam I3 third input type
 * @tparam O  output type
 */
abstract class TernaryModel[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tti3: TypeTag[I3],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[TernaryModel[I1, I2, I3, O]] with OpTransformer3[I1, I2, I3, O]

