/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.sparkwrappers.specific.OpEstimatorWrapper
import enumeratum._
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}

import scala.reflect.runtime.universe.TypeTag

/**
 * OP wrapper for [[org.apache.spark.ml.feature.StringIndexer]]
 *
 * NOTE THAT THIS CLASS EITHER FILTERS OUT OR THROWS AN ERROR IF PREVIOUSLY UNSEEN VALUES APPEAR
 *
 * A label indexer that maps a text column of labels to an ML feature of label indices.
 * The indices are in [0, numLabels), ordered by label frequencies.
 * So the most frequent label gets index 0.
 *
 * @see [[OpIndexToString]] for the inverse transformation
 */
class OpStringIndexer[T <: Text]
(
  uid: String = UID[OpStringIndexer[T]]
)(implicit tti: TypeTag[T])
  extends OpEstimatorWrapper[T, RealNN, StringIndexer, StringIndexerModel](estimator = new StringIndexer(), uid = uid) {

  /**
   * How to handle invalid entries. See [[StringIndexer.handleInvalid]] for more details.
   *
   * @param value StringIndexerHandleInvalid
   * @return this stage
   */
  def setHandleInvalid(value: StringIndexerHandleInvalid): this.type = {
    getSparkMlStage().get.setHandleInvalid(value.entryName.toLowerCase)
    this
  }
}

sealed trait StringIndexerHandleInvalid extends EnumEntry with Serializable

object StringIndexerHandleInvalid extends Enum[StringIndexerHandleInvalid] {
  val values = findValues
  case object Skip extends StringIndexerHandleInvalid
  case object Error extends StringIndexerHandleInvalid
  case object NoFilter extends StringIndexerHandleInvalid
}
