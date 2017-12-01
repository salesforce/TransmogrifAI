/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.Logging

/**
 * Common functionality for classification tests
 */
object FunctionalityForClassificationTests extends Logging {

  /**
   * Calculates cross-entropy, given a dataset
   * @param ds dataset for which cross-entropy is calculated
   * @return the value of cross-entropy. An exception if the dataset is empty.
   */
  def crossEntropyFun(ds: Dataset[(Double, Vector, Vector, Double)]): Double = {
    import ds.sparkSession.implicits._
    if (ds.isEmpty) {
      logWarning("Cannot calculate cross-entropy on an empty dataset")
      0.0
    } else {
      - ds.map { case (lbl, _, prob, _) => math.log(prob.toArray(lbl.toInt)) }.reduce(_ + _)
    }
  }
}
