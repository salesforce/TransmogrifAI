package org.apache.spark.sql.catalyst.csv

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
   * @param rdd           data
   * @param header        CSV header
   * @param options       CSV options
   * @param columnPruning If it is set to true, column names of the requested schema are passed to CSV parser.
   *                      Other column values can be ignored during parsing even if they are malformed.
   * @return inferred schema
   */
  def infer(
    rdd: RDD[Array[String]],
    header: Seq[String],
    options: com.salesforce.op.utils.io.csv.CSVOptions,
    columnPruning: Boolean = true
  ): StructType = {
    val opts = new org.apache.spark.sql.catalyst.csv.CSVOptions(
      parameters = options.copy(header = false).toSparkCSVOptionsMap + ("inferSchema" -> true.toString),
      columnPruning = columnPruning,
      defaultTimeZoneId = "GMT"
    )
    new CSVInferSchema(opts).infer(rdd, header.toArray)
  }

}
