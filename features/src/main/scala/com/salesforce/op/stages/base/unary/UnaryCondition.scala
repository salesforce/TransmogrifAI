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

package com.salesforce.op.stages.base.unary

import com.salesforce.op.UID
import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.features.{FeatureLike, FeatureSparkTypes, OPFeature}
import com.salesforce.op.stages._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.MetadataBuilder
import org.apache.spark.sql.{Dataset, Encoder}

import scala.reflect.runtime.universe.TypeTag


abstract class OpCondition[O1 <: FeatureType, O2 <: FeatureType](val uid: String) extends OpPipelineStageParams {

  def apply(dataset: Dataset[_]): Either[FeatureLike[O1], FeatureLike[O2]]

}

trait ConditionStage {
  self: OpPipelineStageBase =>

  def condition: OpCondition[_ <: FeatureType, _ <: FeatureType]
}

final class UnaryCondition[I <: FeatureType, O1 <: FeatureType, O2 <: FeatureType]
(
  uid: String = UID[UnaryCondition[I, O1, O2]],
  val operationName: String,
  val conditionFn: (FeatureLike[I], Dataset[I#Value]) => Either[FeatureLike[O1], FeatureLike[O2]]
)(implicit val tti: TypeTag[I], val ttiv: TypeTag[I#Value])
  extends OpCondition[O1, O2](uid = uid) {

  type InputFeatures = FeatureLike[I]

  def checkInputLength(features: Array[_]): Boolean = features.length == 1

  protected implicit def inputAsArray(in: InputFeatures): Array[OPFeature] = Array(in)

  def setInput(features: InputFeatures): UnaryCondition[I, O1, O2] = {
    setInputFeatures(features)
    this
  }

  protected def in1: FeatureLike[I] = getInputFeature[I](0).get

  override def copy(extra: ParamMap): UnaryCondition[I, O1, O2] = {
    val that = new UnaryCondition(uid = uid, operationName = operationName, conditionFn = conditionFn)
    copyValues(that, extra)
  }

  // Encoders & converters
  implicit val iEncoder: Encoder[I#Value] = FeatureSparkTypes.featureTypeEncoder[I]
  val iConvert = FeatureTypeSparkConverter[I]()

  def apply(dataset: Dataset[_]): Either[FeatureLike[O1], FeatureLike[O2]] = {
    val df = dataset.select(col(in1.name))
    val ds = df.map(r => iConvert.fromSpark(r.get(0)).value)

    val result = conditionFn(in1, ds)

    val resultStageUid = result match {
      case Left(f) => f.originStage.uid
      case Right(f) => f.originStage.uid
    }

    val meta = new MetadataBuilder().withMetadata(getMetadata())
      .putBoolean("resultIsLeft", result.isLeft)
      .putString("resultStageUid", resultStageUid)

    setMetadata(meta.build())

    result
  }

  def map[U1 <: FeatureType, U2 <: FeatureType]
  (
    left: FeatureLike[O1] => FeatureLike[U1],
    right: FeatureLike[O2] => FeatureLike[U2],
    operationName: String = "mapCondition"
  ): UnaryCondition[I, U1, U2] = {
    new UnaryCondition[I, U1, U2](
      uid = uid,
      operationName = operationName,
      conditionFn = (f, ds) => conditionFn(f, ds) match {
        case Left(f1) => Left(left(f1))
        case Right(f2) => Right(right(f2))
      }
    )
  }

  def asFeature: OPFeature = {
    val stage = new UnaryTransformer[I, I](operationName = operationName, uid = uid) with ConditionStage {
      def transformFn: I => I = identity
      val condition = UnaryCondition.this
    }
    stage.setInput(in1).getOutput()
  }

}

