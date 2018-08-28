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

import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.stages._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Encoder, Encoders}
import org.apache.spark.util.ClosureUtils

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe.TypeTag
import scala.util.Try

/**
 * Takes an input feature of type I1 and a sequence of input features of type I2 and performs
 * a fit operation in order to define a transformation for those (or similar) features. This
 * abstract class should be extended when settable parameters are needed within
 * the fit function
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for input1
 * @param tti2          type tag for input2
 * @param tto           type tag for input
 * @param ttiv1         type tag for input1 value
 * @param ttiv2         type tag for input2 value
 * @param ttov          type tag for output value
 * @tparam I1 input single feature type
 * @tparam I2 input sequence feature type
 * @tparam O output feature type
 */
abstract class BinarySequenceEstimator[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType]
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
) extends Estimator[BinarySequenceModel[I1, I2, O]] with OpPipelineStage2N[I1, I2, O] {

  // Encoders & converters
  implicit val i1Encoder: Encoder[I1#Value] = FeatureSparkTypes.featureTypeEncoder[I1]
  implicit val seqIEncoder: Encoder[Seq[I2#Value]] = Encoders.kryo[Seq[I2#Value]]
  implicit val tupleEncoder = Encoders.tuple[I1#Value, Seq[I2#Value]](i1Encoder, seqIEncoder)
  val convertI1 = FeatureTypeSparkConverter[I1]()
  val seqIConvert = FeatureTypeSparkConverter[I2]()

  /**
   * Function that fits the sequence model
   */
  def fitFn(dataset: Dataset[(I1#Value, Seq[I2#Value])]): BinarySequenceModel[I1, I2, O]

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
  override def fit(dataset: Dataset[_]): BinarySequenceModel[I1, I2, O] = {
    assert(getTransientFeatures.size > 1, "Inputs cannot be empty")
    setInputSchema(dataset.schema).transformSchema(dataset.schema)

    val seqColumns = inN.map(feature => col(feature.name))
    val columns = Array(col(in1.name)) ++ seqColumns

    val df = dataset.select(columns: _*)

    val ds = df.map { r =>
      val rowSeq = r.toSeq
      (convertI1.fromSpark(rowSeq.head).value, rowSeq.tail.map(seqIConvert.fromSpark(_).value))
    }
    val model = fitFn(ds)

    model
      .setParent(this)
      .setInput(in1.asFeatureLike[I1], inN.map(_.asFeatureLike[I2]): _*)
      .setMetadata(getMetadata())
      .setOutputFeatureName(getOutputFeatureName)
  }

}


/**
 * Extend this class and return it from your [[SequenceEstimator]] fit function.
 * Takes a sequence of input features of the same type and produces a single
 * new output feature using the specified function. Performs row wise transformation specified in transformFn.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for input1
 * @param tti2          type tag for input2
 * @param tto           type tag for output
 * @param ttov          type tag for output value
 * @tparam I1 input1 type
 * @tparam I2 input2 type
 * @tparam O output type
 */
abstract class BinarySequenceModel[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType]
(
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[BinarySequenceModel[I1, I2, O]] with OpTransformer2N[I1, I2, O]
