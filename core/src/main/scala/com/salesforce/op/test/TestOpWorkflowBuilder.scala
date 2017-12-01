/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.OPFeature
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * A factory to create [[OpWorkflow]] instances for tests
 */
object TestOpWorkflowBuilder {

  /**
   * Create an instance of a OpWorkflow with a given dataframe and result features
   * @param dataFrame detaframe to run the flow on
   * @param features workflow result features
   * @return
   */
  def apply(dataFrame: DataFrame, features: OPFeature*): OpWorkflow = {
    new OpWorkflow() {
      override def generateRawData()(implicit spark: SparkSession): DataFrame = dataFrame
    }.setResultFeatures(features: _*)
  }

}
