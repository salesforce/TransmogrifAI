/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, _}
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.util.ClosureUtils

import scala.util.Failure

/**
 * Allows columns to be dropped from a feature vector based on properties of the
 * metadata about what is contained in each column (will work only on vectors)
 * created with [[OpVectorMetadata]]
 * @param matchFn function that goes from [[OpVectorColumnMetadata]] to boolean for dropping
 *                columns (cases that evaluate to true will be dropped)
 * @param uid     uid for instance
 */
class DropIndicesByTransformer
(
  val matchFn: OpVectorColumnMetadata => Boolean,
  uid: String = UID[DropIndicesByTransformer]
) extends UnaryTransformer[OPVector, OPVector](operationName = "dropIndicesBy", uid = uid) {

  ClosureUtils.checkSerializable(matchFn) match {
    case Failure(e) => throw new IllegalArgumentException("Provided function is not serializable", e)
    case _ =>
  }

  private lazy val vectorMetadata = OpVectorMetadata(getInputSchema()(in1.name))
  private lazy val columnMetadataToKeep = vectorMetadata.columns.collect { case cm if !matchFn(cm) => cm }
  private lazy val indicesToKeep = columnMetadataToKeep.map { case cm => cm.index }

  override def transformFn: OPVector => OPVector = (v: OPVector) => {
    val vals = new Array[Double](indicesToKeep.length)
    v.value.foreachActive((i, v) => {
      val k = indicesToKeep.indexOf(i)
      if (k >= 0) vals(k) = v
    })
    Vectors.dense(vals).compressed.toOPVector
  }

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val newMetaData = OpVectorMetadata(getOutputFeatureName, columnMetadataToKeep, vectorMetadata.history).toMetadata
    setMetadata(newMetaData)
  }
}
