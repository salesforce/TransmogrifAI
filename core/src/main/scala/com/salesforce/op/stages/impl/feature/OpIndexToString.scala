/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.sparkwrappers.specific.OpTransformerWrapper
import enumeratum._
import org.apache.spark.ml.feature.IndexToString

/**
 * OP wrapper for [[org.apache.spark.ml.feature.IndexToString]]
 *
 * NOTE THAT THIS CLASS EITHER FILTERS OUT OR THROWS AN ERROR IF PREVIOUSLY UNSEEN VALUES APPEAR
 *
 * A transformer that maps a feature of indices back to a new feature of corresponding text values.
 * The index-string mapping is either from the ML attributes of the input feature,
 * or from user-supplied labels (which take precedence over ML attributes).
 *
 * @see [[OpStringIndexer]] for converting text into indices
 */
class OpIndexToString(uid: String = UID[OpIndexToString])
  extends OpTransformerWrapper[RealNN, Text, IndexToString](
    transformer = new IndexToString(), uid = uid
  ) {

  /**
   * Optional array of labels specifying index-string mapping.
   * If not provided or if empty, then metadata from input feature is used instead.
   *
   * @param value array of labels
   * @return
   */
  def setLabels(value: Array[String]): this.type = {
    getSparkMlStage().get.setLabels(value)
    this
  }

  /**
   * Array of labels
   *
   * @return Array of labels
   */
  def getLabels: Array[String] = getSparkMlStage().get.getLabels
}


sealed trait IndexToStringHandleInvalid extends EnumEntry with Serializable

object IndexToStringHandleInvalid extends Enum[IndexToStringHandleInvalid] {
  val values = findValues
  case object NoFilter extends IndexToStringHandleInvalid
  case object Error extends IndexToStringHandleInvalid
}
