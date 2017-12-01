/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.sql.execution.datasources.csv

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType

case object CSVSchemaUtils {

  /**
   * Automatically infer CSV schema from the provided RDD. The process is as follows:
   *
   * Similar to the JSON schema inference:
   *  1. Infer type of each row
   *  2. Merge row types to find common type
   *  3. Replace any null types with string type
   *
   * @param rdd     data
   * @param header  CSV header
   * @param options CSV options
   * @return inferred schema
   */
  def infer(
    rdd: RDD[Array[String]],
    header: Seq[String],
    options: com.salesforce.op.utils.io.csv.CSVOptions
  ): StructType = {
    val opts = new org.apache.spark.sql.execution.datasources.csv.CSVOptions(
      options.copy(header = false).toSparkCSVOptionsMap
    )
    CSVInferSchema.infer(rdd, header.toArray, opts)
  }

}
