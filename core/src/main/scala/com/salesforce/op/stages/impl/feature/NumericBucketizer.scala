/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.numeric.Number
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import enumeratum._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._

import scala.annotation.tailrec
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
  with VectorizerDefaults with NumericBucketizerParams with NumericBucketizerMetadata {

  override def transformFn: I1 => OPVector = input =>
    NumericBucketizer.bucketize[N, I1](
      splits = $(splits),
      trackNulls = $(trackNulls),
      trackInvalid = $(trackInvalid),
      splitInclusion = Inclusion.withNameInsensitive($(splitInclusion)),
      input
    )

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val vectorMeta = makeVectorMetadata(
      input = in1, bucketLabels = $(bucketLabels), trackNulls = $(trackNulls), trackInvalid = $(trackInvalid)
    )
    setMetadata(vectorMeta.toMetadata)
  }
}


trait NumericBucketizerParams extends TrackInvalidParam with TrackNullsParam {

  final val splits = new DoubleArrayParam(
    parent = this, name = "splits",
    doc = "sorted list of split points for bucketizing",
    isValid = NumericBucketizer.checkSplits
  )
  final val bucketLabels = new StringArrayParam(
    parent = this, name = "bucketLabels",
    doc = "sorted list of labels for the buckets"
  )
  final val splitInclusion = new Param[String](
    parent = this, name = "splitInclusion",
    doc = "Should the splits be left or right inclusive. " +
      "Meaning if x1 and x2 are split points, then for Left the bucket interval is [x1, x2) " +
      "and for Right the bucket interval is (x1, x2]."
  )
  setDefault(
    splits -> NumericBucketizer.Splits,
    bucketLabels -> NumericBucketizer.splitsToBucketLabels(NumericBucketizer.Splits, NumericBucketizer.SplitInclusion),
    splitInclusion -> NumericBucketizer.SplitInclusion.toString
  )

  /**
   * Sets the points for bucketizing
   *
   * @param splits       sorted list of split points for bucketizing
   * @param bucketLabels optional sorted list of labels for the buckets
   */
  def setBuckets(splits: Array[Double], bucketLabels: Option[Array[String]] = None): this.type = {
    val theLabels = bucketLabels.getOrElse(NumericBucketizer.splitsToBucketLabels(splits, getSplitInclusion))

    if (theLabels.length != splits.length - 1) {
      throw new IllegalArgumentException("The number of labels should be one less than the number of split points")
    }
    set(this.splits, splits)
    set(this.bucketLabels, theLabels)
  }


  /**
   * Should the splits be left or right inclusive.
   * Meaning if x1 and x2 are split points, then for Left the bucket interval is [x1, x2)
   * and for Right the bucket interval is (x1, x2].
   */
  def setSplitInclusion(v: Inclusion): this.type = {
    // Check if we should update bucket labels as well
    if ($(bucketLabels).sameElements(NumericBucketizer.splitsToBucketLabels($(splits), getSplitInclusion))) {
      set(bucketLabels, NumericBucketizer.splitsToBucketLabels($(splits), v))
    }
    set(splitInclusion, v.toString)
  }

  def getBucketLabels: Array[String] = $(bucketLabels)
  def getSplits: Array[Double] = $(splits)
  def getSplitInclusion: Inclusion = Inclusion.withNameInsensitive($(splitInclusion))
}

private[op] trait NumericBucketizerMetadata {
  self: VectorizerDefaults =>

  protected def makeVectorMetadata(
    input: TransientFeature,
    bucketLabels: Array[String],
    trackInvalid: Boolean,
    trackNulls: Boolean
  ): OpVectorMetadata = {
    val bucketLabelCols = bucketLabels.map(bucketLabel => OpVectorColumnMetadata(
      parentFeatureName = Seq(input.name),
      parentFeatureType = Seq(input.typeName),
      indicatorGroup = Option(input.name),
      indicatorValue = Option(bucketLabel)
    ))
    val trackInvalidCol =
      if (trackInvalid) Seq(OpVectorColumnMetadata(
        parentFeatureName = Seq(input.name),
        parentFeatureType = Seq(input.typeName),
        indicatorGroup = Option(input.name),
        indicatorValue = Some(TransmogrifierDefaults.OtherString)
      ))
      else Nil

    val trackNullCol =
      if (trackNulls) Seq(OpVectorColumnMetadata(
        parentFeatureName = Seq(input.name),
        parentFeatureType = Seq(input.typeName),
        indicatorGroup = Option(input.name),
        indicatorValue = Some(TransmogrifierDefaults.NullString)
      ))
      else Nil

    val finalCols = bucketLabelCols ++ trackInvalidCol ++ trackNullCol
    vectorMetadataFromInputFeatures.withColumns(finalCols)
  }

}


object NumericBucketizer {
  val Splits = Array(Double.NegativeInfinity, 0.0, Double.PositiveInfinity)
  val SplitInclusion = Inclusion.Left

  /**
   * Computes bucket index for given numerical value and one-hot encodes it
   *
   * @param splits         sorted list of split points for bucketizing
   * @param trackNulls     option to keep track of values that were missing
   * @param trackInvalid   option to keep track of invalid values,
   *                       eg. NaN, -/+Inf or values that fall outside the buckets
   * @param splitInclusion should the splits be left or right inclusive
   * @param input          input numerical value
   * @param nev            numeric evidence for double conversion
   * @throws RuntimeException if input value falls outside the bounds of the specified buckets
   * @return one-hot encoded bucket index
   */
  private[op] def bucketize[N, T <: OPNumeric[N]](
    splits: Array[Double],
    trackNulls: Boolean,
    trackInvalid: Boolean,
    splitInclusion: Inclusion,
    input: T
  )(implicit nev: Numeric[N]): OPVector = {
    val numBuckets = splits.length - 1

    /**
     * Computes bucket index for a num value for given splits
     * @return bucket index or -1 if no bucket was found
     */
    def computeBucketIndex(num: Double): Int = {
      @tailrec
      def leftInclusiveIndex(i: Int = 0): Int = {
        if (i >= numBuckets) -1
        else if (splits(i) <= num && num < splits(i + 1)) i
        else leftInclusiveIndex(i + 1)
      }
      @tailrec
      def rightInclusiveIndex(i: Int = 0): Int = {
        if (i >= numBuckets) -1
        else if (splits(i) < num && num <= splits(i + 1)) i
        else rightInclusiveIndex(i + 1)
      }
      splitInclusion match {
        case Inclusion.Left => leftInclusiveIndex()
        case Inclusion.Right => rightInclusiveIndex()
      }
    }

    // One-hot encode the bucket index, while tracking the invalid or error
    val vectorSize = numBuckets + (if (trackInvalid) 1 else 0) + (if (trackNulls) 1 else 0)
    val index: Option[Int] =
      for { v <- input.value } yield {
        val d = nev.toDouble(v)
        val index = computeBucketIndex(d)

        if (trackInvalid) {
          if (index < 0 || !Number.isValid(d)) numBuckets else index
        }
        else {
          if (index < 0) sys.error(s"Numeric value $v falls outside the bounds of the specified buckets") else index
        }
      }

    val (indices, values) = index match {
      case Some(i) => Array(i) -> Array(1.0)
      case None if trackNulls => Array(vectorSize - 1) -> Array(1.0)
      case _ => Array.empty[Int] -> Array.empty[Double]
    }

    Vectors.sparse(vectorSize, indices = indices, values = values).toOPVector
  }

  /**
   * We require splits to be of length >= 3 and to be in strictly increasing order.
   * No NaN split should be accepted.
   */
  private[op] def checkSplits(splits: Array[Double]): Boolean = {
    if (splits.length < 3) return false

    splits.drop(1).foldLeft((splits.head, true)) { case ((prev, valid), curr) =>
      if (prev < curr && !prev.isNaN) (curr, valid) else (prev, false)
    }._2
  }

  /**
   * Computes bucket labels for given splits.
   * E.g splitsToBucketLabels(Array(1,2,3), Inclusion.Left) => Array("[1-2)", "[2-3)")
   *
   * @param splits         split points
   * @param splitInclusion should the splits be left or right inclusive
   * @return bucket labels for given splits
   */
  private[op] def splitsToBucketLabels(splits: Array[Double], splitInclusion: Inclusion): Array[String] = {
    val (prefix, suffix) = splitInclusion match {
      case Inclusion.Left => "[" -> ")"
      case Inclusion.Right => "(" -> "]"
    }
    splits.sliding(2).map { case Array(a, b) => s"$prefix$a-$b$suffix" }.toArray
  }
}

sealed abstract class Inclusion extends EnumEntry with Serializable

object Inclusion extends Enum[Inclusion] {
  val values: Seq[Inclusion] = findValues
  case object Left extends Inclusion
  case object Right extends Inclusion
}

