/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package org.apache.spark.util

import org.apache.spark.sql.functions.{mean, when}
import org.apache.spark.sql.{Column, Dataset, SparkSession}

import scala.util.{Try, Success, Failure}


object SparkUtils {
  private val spark = SparkSession.builder().getOrCreate()
  import spark.implicits._

  /** Preferred alternative to Class.forName(className) */
  def classForName(name: String): Class[_] = Utils.classForName(name)

  def extractDouble(dataset: Dataset[Double]): Double = {
    // Try is necessary because .collect() will fail on an empty Dataset
    // and there doesn't seem to be any other way to check for an empty Dataset,
    // esp. when doing .count > 0 would be very expensive
    Try(dataset.collect().headOption.getOrElse(0.0)) match {
      case Success(value) => value
      case Failure(_) => 0.0
    }
  }

  private def averageCol(dataset: Dataset[_], column: Column): Double = {
    extractDouble(dataset.select(mean(column).as[Double]))
  }

  def averageDoubleCol(dataset: Dataset[Double], column: Column): Double = {
    averageCol(dataset, column)
  }

  def averageFloatCol(dataset: Dataset[Float], column: Column): Double = {
    averageCol(dataset, column)
  }

  def averageIntCol(dataset: Dataset[Int], column: Column): Double = {
    averageCol(dataset, column)
  }

  def averageBoolCol(dataset: Dataset[Boolean], column: Column): Double = {
    averageCol(dataset.select(
      when(column, 1.0).otherwise(0.0).alias(column.toString())
    ), column)
  }
}
