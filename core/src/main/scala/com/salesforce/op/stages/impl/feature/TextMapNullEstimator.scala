/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Creates null indicator columns for a sequence of input TextMap features, originally for use as a separate stage in
 * null tracking for hashed text features (easier to do outside the hashing vectorizer since we can make a null
 * indicator column for each input feature without having to add lots of complex logic in the hashing vectorizer to
 * deal with metadata for shared vs. separate hash spaces.
 */
class TextMapNullEstimator[T <: OPMap[String]]
(
  uid: String = UID[TextMapNullEstimator[_]]
)(implicit tti: TypeTag[T]) extends SequenceEstimator[T, OPVector](
  operationName = "textMapNull", uid = uid) with VectorizerDefaults with MapVectorizerFuns[String, T] {

  protected val shouldCleanValues = true

  override def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {

    val shouldCleanKeys = $(cleanKeys)
    val allKeys: Seq[Seq[String]] = getKeyValues(
      in = dataset,
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues
    )

    // Make the metadata
    val transFeat = getTransientFeatures()
    val colMeta = for {
      (tf, keys) <- transFeat.zip(allKeys)
      key <- keys
    } yield OpVectorColumnMetadata(
      parentFeatureName = Seq(tf.name),
      parentFeatureType = Seq(tf.typeName),
      indicatorGroup = Option(key),
      indicatorValue = Option(OpVectorColumnMetadata.NullString)
    )

    setMetadata(OpVectorMetadata(vectorOutputName, colMeta,
      Transmogrifier.inputFeaturesToHistory(transFeat, stageName))
      .toMetadata)

    new TextMapNullModel[T](allKeys, shouldCleanKeys, shouldCleanValues,
      operationName = operationName, uid = uid)
  }
}

final class TextMapNullModel[T <: OPMap[String]] private[op]
(
  val allKeys: Seq[Seq[String]],
  val cleanKeys: Boolean,
  val cleanValues: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults with CleanTextMapFun {

  def transformFn: Seq[T] => OPVector = row => {
    row.zipWithIndex.flatMap {
      case (map, i) =>
        val keys = allKeys(i)
        val cleaned = cleanMap(map.v, shouldCleanKey = cleanKeys, shouldCleanValue = cleanValues)

        keys.map(k => if (cleaned.contains(k)) 0.0 else 1.0)
    }.toOPVector
  }

}
