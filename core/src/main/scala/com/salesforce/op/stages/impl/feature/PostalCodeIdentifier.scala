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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import org.apache.spark.ml.param.{DoubleParam, ParamValidators}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}

import scala.io.Source
import scala.util.Try

class PostalCodeIdentifier
(
  uid: String = UID[PostalCodeIdentifier],
  operationName: String = "postal code identifier"
) extends UnaryEstimator[Text, PostalCodeMap](
  uid = uid,
  operationName = operationName
) {
  // Parameters
  val defaultThreshold = new DoubleParam(
    parent = this,
    name = "defaultThreshold",
    doc = "default fraction of successful postal code validations before treating as Postal Code",
    isValid = (value: Double) => {
      ParamValidators.gt(0.0)(value) && ParamValidators.lt(1.0)(value)
    }
  )
  setDefault(defaultThreshold, 0.90)


  def setThreshold(value: Double): this.type = set(defaultThreshold, value)

  lazy private val postalCodeDictionary = {
    val postalCodeDictionary = collection.mutable.Map.empty[String, (Option[Double], Option[Double])]
    val dictionaryPath = "/USPostalCodes.txt"
    val stream = getClass.getResourceAsStream(dictionaryPath)
    val buffer = Source.fromInputStream(stream)
    for {row <- buffer.getLines} {
      val cols = row.split(",").map(_.trim)
      val code = cols(0)
      val lat = Try { cols(1).toDouble }.toOption
      val lng = Try { cols(2).toDouble }.toOption
      postalCodeDictionary += (code -> (lat, lng))
    }
    buffer.close
    postalCodeDictionary
  }

  private def extractDouble(dataset: DataFrame): Double = dataset.collect().headOption.getOrElse(Row(0.0)).getDouble(0)

  private def averageBoolCol(dataset: Dataset[Boolean], column: Column): Double = {
    extractDouble(dataset.select(mean(column.cast("integer"))))
  }

  private def attemptToExtractPostalCode(dataset: Dataset[Text#Value], column: Column): Dataset[Text#Value] = {
    // Regex for all types of postal codes: ^\\d{5}(?:[-\\s]\\d{4})?$
    dataset.withColumn(
      column.toString, regexp_extract(column, "^\\d{5}$", 0)
    ).asInstanceOf[Dataset[Text#Value]]
  }

  private def guardChecks(dataset: Dataset[Text#Value], column: Column): Boolean = {
    averageBoolCol(dataset.withColumn(
      column.toString, column notEqual ""
    ).asInstanceOf[Dataset[Boolean]], column) > $(defaultThreshold)
  }

  private def dictCheck: UserDefinedFunction = udf((s: String) => {
    postalCodeDictionary contains s
  }: Boolean)

  private def predictIfPostalCode(dataset: Dataset[Text#Value], column: Column): Dataset[Boolean] = {
    dataset.select(dictCheck(column).alias(column.toString)).asInstanceOf[Dataset[Boolean]]
  }

  def fitFn(dataset: Dataset[Text#Value]): PostalCodeIdentifierModel = {
    assert(dataset.schema.fieldNames.length == 1)
    val column = col(dataset.schema.fieldNames.head)
    val matches = attemptToExtractPostalCode(dataset, column)
    if (
      guardChecks(matches, column) &&
      averageBoolCol(predictIfPostalCode(matches, column), column) >= $(defaultThreshold)
    ) {
      new PostalCodeIdentifierModel(uid, true)
    } else new PostalCodeIdentifierModel(uid, false)
  }
}

class PostalCodeIdentifierModel(override val uid: String, val treatAsPostalCode: Boolean)
  extends UnaryModel[Text, PostalCodeMap]("postal code identifier", uid) {
  lazy private val postalCodeDictionary = {
    val postalCodeDictionary = collection.mutable.Map.empty[String, (Option[Double], Option[Double])]
    val dictionaryPath = "/USPostalCodes.txt"
    val stream = getClass.getResourceAsStream(dictionaryPath)
    val buffer = Source.fromInputStream(stream)
    for {row <- buffer.getLines} {
      val cols = row.split(",").map(_.trim)
      val code = cols(0)
      val lat = Try { cols(1).toDouble }.toOption
      val lng = Try { cols(2).toDouble }.toOption
      postalCodeDictionary += (code -> (lat, lng))
    }
    buffer.close
    postalCodeDictionary
  }
  def transformFn: Text => PostalCodeMap = input => {
    val postalCode = input.value.getOrElse("")
    if (treatAsPostalCode) {
      val (latOption, lngOption) = postalCodeDictionary.getOrElse(postalCode, (None, None))
      (latOption, lngOption) match {
        case (Some(lat), Some(lng)) =>
          PostalCodeMap(Map(postalCode -> "true", "lat" -> lat.toString, "lng" -> lng.toString))
        case _ => PostalCodeMap(Map(postalCode -> "true", "lat" -> "", "lng" -> ""))
      }
    }
    else PostalCodeMap(Map.empty[String, String])
  }
}
