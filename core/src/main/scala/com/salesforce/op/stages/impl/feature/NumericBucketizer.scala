/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.spark.OpVectorColumnMetadata
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{DoubleArrayParam, Params, StringArrayParam}

import scala.reflect.runtime.universe.TypeTag

/**
 * Numeric Bucketizer
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti1          type tag for numeric feature type
 * @param nev           numeric evidence for feature type value
 * @tparam N  numeric feature type value
 * @tparam I1 numeric feature type
 */
class NumericBucketizer[N, I1 <: OPNumeric[N]]
(
  operationName: String = "numBuck",
  uid: String = UID[NumericBucketizer[_, _]]
)(
  implicit val tti1: TypeTag[I1],
  val nev: Numeric[N]
) extends UnaryTransformer[I1, OPVector](operationName = operationName, uid = uid)
  with VectorizerDefaults with NumericBucketizerParams with TrackNullsParam {

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val cols = $(bucketLabels).map(label =>
      OpVectorColumnMetadata(
        parentFeatureName = Seq(in1.name),
        parentFeatureType = Seq(in1.typeName),
        indicatorGroup = Option(in1.name),
        indicatorValue = Option(label)
      )
    )
    val finalCols = if ($(trackNulls)) {
      cols :+ OpVectorColumnMetadata(
        parentFeatureName = Seq(in1.name),
        parentFeatureType = Seq(in1.typeName),
        indicatorGroup = Option(in1.name),
        indicatorValue = Some(TransmogrifierDefaults.NullString)
      )
    } else cols

    setMetadata(vectorMetadataFromInputFeatures.withColumns(finalCols).toMetadata)
  }

  override def transformFn: I1 => OPVector = input => {
    val theSplits = $(splits)
    val numBuckets = theSplits.length - 1

    def error = throw new Exception(s"Numeric value $input falls outside the bounds of the specified buckets")

    def getArrays(num: Double): Array[Double] = {
      val index = theSplits.indexWhere(split => split > num) match {
        case -1 => if (theSplits(numBuckets) == num) numBuckets else error
        case 0 => error
        case x => x
      }
      oneHot(index, numBuckets)
    }

    val buckets = input.value.map(v => getArrays(nev.toDouble(v))).getOrElse(Array.fill(numBuckets)(0.0))
    val finalBuckets = if ($(trackNulls)) buckets :+ (input.isEmpty: Double) else buckets

    Vectors.dense(finalBuckets).toOPVector
  }
}


// TODO: Spark Bucketizer has a SKIP_INVALID param. Should we support it? What should it do?
trait NumericBucketizerParams extends Params {
  final val splits = new DoubleArrayParam(
    parent = this, name = "splits", doc = "sorted list of split points for bucketizing. Splits are left inclusive," +
      "meaning if x1 and x2 are split points, then the bucket interval is [x1, x2). ",
    isValid = NumericBucketizer.checkSplits
  )

  setDefault(splits, NumericBucketizer.Splits)

  final val bucketLabels = new StringArrayParam(
    parent = this, name = "bucketLabels", doc = "sorted list of labels for the buckets"
  )

  setDefault(bucketLabels, Array[String]("-Inf_0", "0_Inf"))

  def setBuckets(splits: Array[Double], bucketLabels: Option[Array[String]] = None): this.type = {
    val theLabels = bucketLabels.getOrElse(splits.sliding(2).map { case Array(a, b) => s"$a-$b" }.toArray)

    if (theLabels.length != splits.length - 1) {
      throw new IllegalArgumentException("The number of labels should be one less than the number of split points")
    }
    set(this.splits, splits)
    set(this.bucketLabels, theLabels)
  }

}

object NumericBucketizer {

  val Splits = Array(Double.NegativeInfinity, 0.0, Double.PositiveInfinity)

  /**
   * We require splits to be of length >= 3 and to be in strictly increasing order.
   * No NaN split should be accepted.
   */
  def checkSplits(splits: Array[Double]): Boolean = {
    if (splits.length < 3) return false

    splits.drop(1).foldLeft((splits.head, true)) { case ((prev, incr), curr) =>
      if (prev < curr && !prev.isNaN) (curr, incr) else (prev, false)
    }._2
  }

}

