/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * A label indexer that maps a text column of labels to an ML feature of label indices.
 * The indices are in [0, numLabels), ordered by label frequencies.
 * So the most frequent label gets index 0.
 *
 * @see [[OpIndexToStringNoFilter]] for the inverse transformation
 */
class OpStringIndexerNoFilter[I <: Text]
(
  uid: String = UID[OpStringIndexerNoFilter[I]]
)(implicit tti: TypeTag[I], ttiv: TypeTag[I#Value])
  extends UnaryEstimator[I, RealNN](operationName = "str2idx", uid = uid) with SaveOthersParams {

  setDefault(unseenName, OpStringIndexerNoFilter.UnseenNameDefault)

  def fitFn(data: Dataset[I#Value]): UnaryModel[I, RealNN] = {
    val unseen = $(unseenName)
    val counts = data.rdd.countByValue()
    val labels = counts.toSeq.sortBy(-_._2).map(_._1).toArray
    val otherPos = labels.length

    val cleanedLabels = labels.map(_.getOrElse("null")) :+ unseen
    val metadata = NominalAttribute.defaultAttr.withName(outputName).withValues(cleanedLabels).toMetadata()
    setMetadata(metadata)

    new OpStringIndexerNoFilterModel[I](labels, otherPos, operationName = operationName, uid = uid)
  }
}

final class OpStringIndexerNoFilterModel[I <: Text] private[op]
(
  val labels: Seq[Option[String]],
  val otherPos: Int,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[I]) extends UnaryModel[I, RealNN](operationName = operationName, uid = uid) {

  private val labelsMap = labels.zipWithIndex.toMap
  def transformFn: I => RealNN = in => labelsMap.getOrElse(in.value, otherPos).toRealNN
}

object OpStringIndexerNoFilter {
  val UnseenNameDefault = "UnseenLabel"
}
