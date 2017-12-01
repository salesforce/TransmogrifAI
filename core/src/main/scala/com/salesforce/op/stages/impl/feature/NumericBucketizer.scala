/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.spark.OpVectorColumnMetadata
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{DoubleArrayParam, Params, StringArrayParam}

import scala.reflect.runtime.universe.TypeTag

/**
 * Numeric Bucketizer
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1
 * @param nev
 * @tparam N
 * @tparam I1
 */
class NumericBucketizer[N, I1 <: OPNumeric[N]]
(
  operationName: String = "numBuck",
  uid: String = UID[NumericBucketizer[_, _]]
)(
  implicit val tti1: TypeTag[I1],
  val nev: Numeric[N]
) extends UnaryTransformer[I1, OPVector](operationName = operationName, uid = uid)
  with VectorizerDefaults with NumericBucketizerParams {

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val cols = $(labels).map(label =>
      OpVectorColumnMetadata(
        parentFeatureName = Seq(in1.name),
        parentFeatureType = Seq(in1.typeName),
        indicatorGroup = Option(in1.name),
        indicatorValue = Option(label)
      )
    )
    setMetadata(vectorMetadataFromInputFeatures.withColumns(cols).toMetadata)
  }

  override def transformFn: I1 => OPVector = dataIn => {
    val theSplits = $(splits)
    val numBuckets = theSplits.length - 1
    def error = throw new Exception(s"numeric value $dataIn falls outside the bounds of the specified buckets")

    def getVectors(num: Double): Vector = {
      val index = theSplits.indexWhere(split => split > num) match {
        case -1 => if (theSplits(numBuckets) == num) numBuckets else error
        case 0 => error
        case x => x
      }
      Vectors.dense(oneHot(index, numBuckets))
    }
    val vector =
      dataIn.value.map(v => getVectors(nev.toDouble(v)))
        .getOrElse(Vectors.dense(Array.fill(numBuckets)(0.0)))

    vector.toOPVector
  }
}


// TODO: Spark Bucketizer has a SKIP_INVALID param. Should we support it? What should it do?
trait NumericBucketizerParams extends Params {
  final val splits = new DoubleArrayParam(
    parent = this, name = "splits", doc = "sorted list of split points for bucketizing",
    isValid = checkSplits
  )

  setDefault(splits, Array[Double](Double.NegativeInfinity, 0, Double.PositiveInfinity))

  final val labels = new StringArrayParam(
    parent = this, name = "bucketLabels", doc = "sorted list of labels for the buckets"
  )

  setDefault(labels, Array[String]("-Inf_0", "0_Inf"))

  def setBuckets(splitPoints: Array[Double], bucketLabels: Option[Array[String]] = None): this.type = {
    val theLabels = bucketLabels
      .getOrElse(splitPoints.sliding(2).map { case (Array(a, b)) => s"${a.toString}-${b.toString}" }.toArray)

    if (theLabels.length != splitPoints.length - 1) {
      throw new IllegalArgumentException("The number of labels should be one less than the number of split points")
    }
    set(splits, splitPoints)
    set(labels, theLabels)
  }


  /**
   * We require splits to be of length >= 3 and to be in strictly increasing order.
   * No NaN split should be accepted.
   */
  private def checkSplits(splits: Array[Double]): Boolean = {
    if (splits.length < 3) return false

    splits.drop(1).foldLeft((splits.head, true)) { case ((prev, incr), curr) =>
      if (prev < curr && !prev.isNaN) (curr, incr) else (prev, false)
    }._2
  }

}

