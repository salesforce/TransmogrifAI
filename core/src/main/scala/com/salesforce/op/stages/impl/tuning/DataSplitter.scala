/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.UID
import com.salesforce.op.stages.impl.tuning.SelectorData.LabelFeaturesKey
import org.apache.spark.ml.param._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.MetadataBuilder

case object DataSplitter {

  /**
   * Creates instance that will split data into training and test set
   *
   * @param seed                set for the random split
   * @param reserveTestFraction fraction of the data used for test
   * @return data splitter
   */
  def apply(
    seed: Long = SplitterParamsDefault.seedDefault,
    reserveTestFraction: Double = SplitterParamsDefault.ReserveTestFractionDefault
  ): DataSplitter = {
    new DataSplitter()
      .setSeed(seed)
      .setReserveTestFraction(reserveTestFraction)
  }
}

/**
 * Instance that will only split the data
 *
 * @param uid
 */
private[op] class DataSplitter(uid: String = UID[DataSplitter]) extends Splitter(uid = uid) {

  final override def prepare(data: Dataset[LabelFeaturesKey]): ModelData =
    new ModelData(data, new MetadataBuilder())

  final override def copy(extra: ParamMap): DataSplitter = {
    val copy = new DataSplitter(uid)
    copyValues(copy, extra)
  }
}
