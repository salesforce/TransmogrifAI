/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.io.csv

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * CSV wrapper around spark.read.csv
 *
 * @param options CSV options
 */
class CSVInOut(val options: CSVOptions) {

  /**
   * Method for reading CSV file into an DataFrame
   *
   * @param path path to file
   * @return DataFrame
   */
  def readDataFrame(path: String)(implicit spark: SparkSession): DataFrame =
    spark.read.options(options.toSparkCSVOptionsMap).csv(path)

  /**
   * Method for reading CSV file into an RDD of columns (a collections of strings)
   *
   * @param path  path to file
   * @param spark spark session
   * @return RDD of columns (a collections of strings)
   */
  def readRDD(path: String)(implicit spark: SparkSession): RDD[Seq[String]] = {
    import spark.implicits._
    readDataFrame(path).map(row =>
      row.toSeq.collect { case null => null; case v => v.toString }
    ).rdd
  }

}

/**
 * CSV options
 *
 * @param separator   column separator
 * @param quoteChar   quote  character
 * @param escapeChar  the escape character to use
 * @param allowEscape allow the specification of a particular escape character
 * @param header      first line is a header
 */
case class CSVOptions
(
  separator: String = ",",
  quoteChar: String = "\"",
  escapeChar: String = "\\",
  allowEscape: Boolean = false,
  header: Boolean = false
) {

  /**
   * Create a Map matching [[org.apache.spark.sql.execution.datasources.csv.CSVOptions]] structure
   *
   * @return Map matching [[org.apache.spark.sql.execution.datasources.csv.CSVOptions]] structure
   */
  def toSparkCSVOptionsMap: Map[String, String] = Map(
    "sep" -> separator,
    "quote" -> quoteChar,
    "escape" -> escapeChar,
    "escapeQuotes" -> allowEscape.toString,
    "header" -> header.toString
  )

}
