/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.SequenceAggregators
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{BooleanParam, DoubleParam}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Converts a sequence of Nullable Numeric features into a vector feature.
 * Can choose to fill null values with the mean or a constant
 *
 * @param uid uid for instance
 */
class RealVectorizer[T <: Real]
(
  uid: String = UID[RealVectorizer[_]],
  operationName: String = "vecReal"
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends SequenceEstimator[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults with TrackNullsParam {

  final val fillValue = new DoubleParam(this, "fillValue", "default value for FillWithConstant")
  setDefault(fillValue, 0.0)

  final val withConstant = new BooleanParam(this, "fillWithConstant",
    "boolean to check if filling the nulls with a constant value")

  setDefault(withConstant, true)


  def setFillWithConstant(value: Double): this.type = {
    set(fillValue, value)
    set(withConstant, true)
  }

  def setFillWithMean: this.type = {
    set(withConstant, false)
  }

  private def constants(): Seq[Double] = {
    val size = getInputFeatures().length
    val defValue = $(fillValue)
    val constants = List.fill(size)(defValue)
    constants
  }

  private def means(dataset: Dataset[Seq[T#Value]]): Seq[Double] = {
    val size = getInputFeatures().length
    val means = dataset.select(SequenceAggregators.MeanSeqNullNum(size)).first()
    means
  }

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    if ($(trackNulls)) setMetadata(vectorMetadataWithNullIndicators.toMetadata)

    val fillValues = if ($(withConstant)) constants() else means(dataset)
    new RealVectorizerModel[T](
      fillValues = fillValues, trackNulls = $(trackNulls), operationName = operationName, uid = uid)
  }

}

private final class RealVectorizerModel[T <: Real]
(
  val fillValues: Seq[Double],
  val trackNulls: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults {

  def transformFn: Seq[T] => OPVector = row => {
    val replaced =
      if (!trackNulls) {
        row.zip(fillValues).
          map { case (r, m) => r.value.getOrElse(m) }
      }
      else {
        row.zip(fillValues).
          flatMap { case (r, m) => r.value.getOrElse(m) :: booleanToDouble(r.isEmpty) :: Nil }
      }
    Vectors.dense(replaced.toArray).toOPVector
  }

}
