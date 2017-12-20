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
import org.apache.spark.ml.param.{BooleanParam, LongParam}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Converts a sequence of Integral features into a vector feature.
 * Can choose to fill null values with the mean or a constant
 *
 * @param uid uid for instance
 */
class IntegralVectorizer[T <: Integral]
(
  uid: String = UID[IntegralVectorizer[_]],
  operationName: String = "vecInt"
) (implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends SequenceEstimator[T, OPVector](operationName = operationName, uid = uid)
  with VectorizerDefaults with TrackNullsParam {

  final val fillValue = new LongParam(this, "fillValue", "default value for FillWithConstant")
  setDefault(fillValue, 0L)

  final val withConstant = new BooleanParam(this, "fillWithConstant",
    "boolean to check if filling the nulls with a constant value")
  setDefault(withConstant, true)

  def setFillWithConstant(value: Long): this.type = {
    set(fillValue, value)
    set(withConstant, true)
  }
  def setFillWithMode: this.type = set(withConstant, false)

  private def constants(): Seq[Long] = {
    val size = getInputFeatures().length
    val defValue = $(fillValue)
    val constants = List.fill(size)(defValue)
    constants
  }

  private def mode(dataset: Dataset[Seq[T#Value]]): Seq[Long] = {
    val size = getInputFeatures().length
    dataset.select(SequenceAggregators.ModeSeqNullInt(size = size).toColumn).first()
  }

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    if ($(trackNulls)) setMetadata(vectorMetadataWithNullIndicators.toMetadata)

    val fillValues = if ($(withConstant)) constants() else mode(dataset)

    new IntegralVectorizerModel(
      fillValues = fillValues, trackNulls = $(trackNulls), operationName = operationName, uid = uid)

  }

}

private final class IntegralVectorizerModel[T <: Integral]
(
  val fillValues: Seq[Long],
  val trackNulls: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults {

  def transformFn: Seq[T] => OPVector = row => {
    val replaced = if (!trackNulls) {
      row.zip(fillValues).
        map { case (i, m) => i.value.getOrElse(m).toDouble }
    }
    else {
      row.zip(fillValues).
        flatMap { case (i, m) => i.value.getOrElse(m).toDouble :: booleanToDouble(i.value.isEmpty) :: Nil }
    }
    Vectors.dense(replaced.toArray).toOPVector
  }

}
