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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.json.{JsonLike, JsonUtils}
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

import scala.reflect.runtime.universe.TypeTag
import scala.util.{Failure, Try}


/**
 * A trait extended by a class class containing the args needed to define a family of scaling & descaling functions
 */
trait ScalingArgs extends JsonLike

/**
 * Case class for Scaling families that take no parameters
 */
case class EmptyArgs() extends ScalingArgs

/**
 * Parameters need to uniquely define a linear scaling function
 *
 * @param slope     the slope of the linear scaler
 * @param intercept the x axis intercept of the linear scaler
 */
case class LinearScalerArgs(slope: Double, intercept: Double) extends ScalingArgs

/**
 * Trait for defining a new family of scaling functions
 * scalingType: a ScalingType Enum for the scaling name
 * args: A case class containing the args needed to define scaling and inverse scaling functions
 * scale: The scaling function
 * descale: The inverse scaling function
 *
 * To add a new family of scaling functions: Add an entry to the scalingType enum, define a Case class extending Scaler,
 * and add a case statement to both the Scaler and ScalerMetaData case classes
 */
trait Scaler extends Serializable {
  def scalingType: ScalingType
  def args: ScalingArgs
  def scale(v: Double): Double
  def descale(v: Double): Double
}

/**
 * [[Scaler]] instance factory
 */
object Scaler {
  /**
   * Creates a new instance of a [[Scaler]] based on specified [[ScalingType]] and [[ScalingArgs]]
   * @param scalingType desired scaling type
   * @param args specific scaling args
   * @return
   */
  def apply(scalingType: ScalingType, args: ScalingArgs): Scaler = (scalingType, args) match {
    case (ScalingType.Linear, l: LinearScalerArgs) => LinearScaler(l)
    case (ScalingType.Logarithmic, _) => LogScaler()
    case (t, args) => throw new IllegalArgumentException(
      s"Invalid combination of scaling type '$t' and args type '${args.getClass.getSimpleName}'")
  }
}

/**
 * A case class representing a logarithmic scaling function
 */
case class LogScaler() extends Scaler {
  val scalingType: ScalingType = ScalingType.Logarithmic
  val args: ScalingArgs = EmptyArgs()
  def scale(v: Double): Double = math.log(v)
  def descale(v: Double): Double = math.exp(v)
}

/**
 * A case class representing a linear scaling function
 *
 * @param args case class containing the slope and intercept of the scaling function
 */
case class LinearScaler(args: LinearScalerArgs) extends Scaler {
  require(args.slope != 0.0, "LinearScaler must have a non-zero slope to be invertible")
  val scalingType: ScalingType = ScalingType.Linear

  def scale(v: Double): Double = args.slope * v + args.intercept

  def descale(v: Double): Double = (v - args.intercept) / args.slope
}

/**
 * Metadata containing the info needed to reconstruct a Scaler instance
 *
 * @param scalingType the family of functions containing the scaler
 * @param scalingArgs the args uniquely defining a function in the scaling family
 */
case class ScalerMetadata(scalingType: ScalingType, scalingArgs: ScalingArgs) {
  def toMetadata(): Metadata = new MetadataBuilder()
    .putString(ScalerMetadata.scalingTypeName, scalingType.entryName)
    .putString(ScalerMetadata.scalingArgsName, scalingArgs.toJson(pretty = false))
    .build()
}

object ScalerMetadata extends {

  val scalingTypeName = "scalingType"
  val scalingArgsName = "scalingArgs"

  def apply(meta: Metadata): Try[ScalerMetadata] = for {
    scalingType <- Try(ScalingType.withName(meta.getString(scalingTypeName)))
    args <- Try(meta.getString(scalingArgsName))
    meta <- scalingType match {
      case t@ScalingType.Linear =>
        JsonUtils.fromString[LinearScalerArgs](args).map(ScalerMetadata(t, _))
      case t@ScalingType.Logarithmic =>
        JsonUtils.fromString[EmptyArgs](args).map(ScalerMetadata(t, _))
      case t =>
        Failure(new IllegalArgumentException(s"Unsupported scaling type $t"))
    }
  } yield meta

}

/**
 * Scaling transformer that applies a scaling function to a numerical feature
 *
 * @param uid         uid for instance
 * @param scalingType type of scaling functions
 * @param scalingArgs arguments to define the scaling function
 * @param tti         type tag for input
 * @param tto         type tag for output
 * @param ttov        type tag for output value
 * @tparam I input feature type
 * @tparam O output feature type
 */
final class ScalerTransformer[I <: Real, O <: Real]
(
  uid: String = UID[ScalerTransformer[_, _]],
  val scalingType: ScalingType,
  val scalingArgs: ScalingArgs
)(implicit tti: TypeTag[I], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends UnaryTransformer[I, O](operationName = "scaler", uid = uid) {

  private val ftFactory = FeatureTypeFactory[O]()
  private val scaler = Scaler(scalingType, scalingArgs)

  def transformFn: I => O = v => {
    val scaled = v.toDouble.map(scaler.scale)
    ftFactory.newInstance(scaled)
  }

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val meta = ScalerMetadata(scalingType, scalingArgs).toMetadata()
    setMetadata(meta)
  }
}


