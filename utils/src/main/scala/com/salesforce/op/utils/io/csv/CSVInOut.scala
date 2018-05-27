/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
  def readDataFrame(path: String)(implicit spark: SparkSession): DataFrame = {
    val reader = spark.read.options(options.toSparkCSVOptionsMap).format(options.format)
    reader.load(path)
  }

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
 * @param format      specifies the input data source format. For more info see [[org.apache.spark.sql.DataFrameReader]]
 */
case class CSVOptions
(
  separator: String = ",",
  quoteChar: String = "\"",
  escapeChar: String = "\\",
  allowEscape: Boolean = false,
  header: Boolean = false,
  format: String = "csv"
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
