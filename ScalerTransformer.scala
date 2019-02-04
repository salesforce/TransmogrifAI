package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryTransformer
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.UID
import com.salesforce.op.utils.json.{JsonLike, JsonUtils}

import org.apache.spark.sql.types.{Metadata, MetadataBuilder}
import scala.reflect.runtime.universe.TypeTag
import scala.util.{Failure, Success, Try}




trait ScalingArgs extends JsonLike
case class EmptyArgs() extends ScalingArgs
case class LinearScalerArgs(slope: Double, intercept: Double) extends ScalingArgs

trait Scaler extends Serializable {
  def scalingType: ScalingType
  def args: ScalingArgs
  def scale(v: Double): Double
  def descale(v: Double): Double
}

object Scaler {
  def apply(scalingType: ScalingType, args: ScalingArgs): Scaler = (scalingType, args) match {
    case (ScalingType.Linear, l: LinearScalerArgs) => LinearScaler(l)
    case (ScalingType.Logarithmic, _) => LogScaler()
    case (t, args) => throw new IllegalArgumentException(
      s"Invalid combination of scaling type '$t' and args type '${args.getClass.getSimpleName}'")
  }
}

case class LogScaler() extends Scaler {
  val scalingType: ScalingType = ScalingType.Logarithmic
  val args: ScalingArgs = EmptyArgs()
  def scale(v: Double): Double = math.log(v)
  def descale(v: Double): Double = math.exp(v)
}

case class LinearScaler(args: LinearScalerArgs) extends Scaler {
  require(args.slope != 0.0, "Must have a non zero slope to be invertible")
  val scalingType: ScalingType = ScalingType.Linear
  def scale(v: Double): Double = args.slope * v + args.intercept
  def descale(v: Double): Double = (v - args.intercept) / args.slope
}

case class ScalerMetadata(scalingType: ScalingType, scalingArgs: ScalingArgs) {
  def toMetadata(): Metadata = new MetadataBuilder()
    .putString("scalingType", scalingType.entryName)
    .putString("scalingArgs", scalingArgs.toJson(pretty = false))
    .build()
}

object ScalerMetadata extends {
  def apply(meta: Metadata): Try[ScalerMetadata] = for {
    scalingType <- Try(ScalingType.withName(meta.getString("scalingType")))
    args <- Try(meta.getString("scalingArgs"))
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

/**
 *  - 1st input feature is the label feature to descale
 *  - 2nd input feature is the scaled label to get the metadata from
 */
final class DescalerTransformer[I1 <: Real, I2 <: Real, O <: Real]
(
  uid: String = UID[DescalerTransformer[_, _, _]]
)(implicit tti1: TypeTag[I1], tti2: TypeTag[I2], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends BinaryTransformer[I1, I2, O](operationName = "descaler", uid = uid) {

  private val ftFactory = FeatureTypeFactory[O]()

  @transient private lazy val meta = getInputSchema()(in2.name).metadata
  @transient private lazy val scalerMeta: ScalerMetadata = ScalerMetadata(meta) match {
    case Success(sm) => sm
    case Failure(error) =>
      throw new RuntimeException(s"Failed to extract scaler metadata for input feature '${in2.name}'", error)
  }
  @transient private lazy val scaler = Scaler(scalerMeta.scalingType, scalerMeta.scalingArgs)

  def transformFn: (I1, I2) => O = (v, _) => {
    val descaled = v.toDouble.map(scaler.descale)
    ftFactory.newInstance(descaled)
  }

}

/**
 *  - 1st input feature is the prediction feature to descale
 *  - 2nd input feature is the scaled label to get the metadata from
 */
final class PredictionDescaler[I <: Real, O <: Real]
(
  uid: String = UID[DescalerTransformer[_, _, _]]
)(implicit tti2: TypeTag[I], tto: TypeTag[O], ttov: TypeTag[O#Value])
  extends BinaryTransformer[Prediction, I, O](operationName = "descaler", uid = uid) {

  private val ftFactory = FeatureTypeFactory[O]()

  @transient private lazy val meta = getInputSchema()(in2.name).metadata
  @transient private lazy val scalerMeta: ScalerMetadata = ScalerMetadata(meta) match {
    case Success(sm) => sm
    case Failure(error) =>
      throw new RuntimeException(s"Failed to extract scaler metadata for input feature '${in2.name}'", error)
  }
  @transient private lazy val scaler = Scaler(scalerMeta.scalingType, scalerMeta.scalingArgs)

  def transformFn: (Prediction, I) => O = (v, _) => {
    val descaled = scaler.descale(v.prediction)
    ftFactory.newInstance(descaled)
  }
}
