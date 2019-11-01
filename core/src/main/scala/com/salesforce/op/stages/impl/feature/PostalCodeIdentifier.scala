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
import com.salesforce.op.utils.text.TextUtils.getBestRegexMatch
import org.apache.spark.ml.param.{DoubleParam, ParamValidators}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.util.SparkUtils.averageBoolCol

import scala.collection.mutable
import scala.io.Source
import scala.reflect.runtime.universe.TypeTag
import scala.util.Try
import scala.util.matching.Regex

trait PostalCodeHelpers {
  lazy val postalCodeDictionary: mutable.Map[String, (Option[Double], Option[Double])] = {
    val postalCodeDictionary = collection.mutable.Map.empty[String, (Option[Double], Option[Double])]
    val dictionaryPath = "/USPostalCodes.txt"
    val stream = getClass.getResourceAsStream(dictionaryPath)
    val buffer = Source.fromInputStream(stream)
    for {row <- buffer.getLines} {
      val cols = row.split(",").map(_.trim)
      val code = cols(0)
      val lat = Try {
        cols(1).toDouble
      }.toOption
      val lng = Try {
        cols(2).toDouble
      }.toOption
      postalCodeDictionary += (code -> (lat, lng))
    }
    buffer.close
    postalCodeDictionary
  }
  val patterns: Seq[Regex] = Seq(
    ".*(\\d{5}).*".r,
    ".*(\\d{4}).*".r,
    ".*(\\d{3}).*".r
  )

  def findBestPostalCodeMatch(s: String): String = {
    val result = getBestRegexMatch(patterns, s)
    // Pad result with leading zeros if needed
    if (result.length < 5) {
      val numMissingDigits = 5 - result.length
      (Seq.fill(numMissingDigits)("0") :+ result).mkString("")
    }
    else result
  }
}

class PostalCodeIdentifier[T <: Text]
(
  uid: String = UID[PostalCodeIdentifier[_]],
  operationName: String = "postal code identifier"
)
(
  implicit tti: TypeTag[T],
  override val ttiv: TypeTag[T#Value]
) extends UnaryEstimator[T, PostalCodeMap](
  uid = uid,
  operationName = operationName
) with PostalCodeHelpers {
  private val spark = SparkSession.builder().getOrCreate()
  import spark.implicits._
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

  private def checkIfPostalCode: UserDefinedFunction = udf((s: String) => {
    val matched = findBestPostalCodeMatch(s)
    matched != "" && (postalCodeDictionary contains matched)
  }: Boolean)

  def fitFn(dataset: Dataset[Text#Value]): PostalCodeIdentifierModel[T] = {
    assert(dataset.schema.fieldNames.length == 1)
    val column = col(dataset.schema.fieldNames.head)
    if (
      averageBoolCol(
        dataset.select(checkIfPostalCode(column).alias(column.toString).as[Boolean]),
        column
      ) >= $(defaultThreshold)
    ) {
      new PostalCodeIdentifierModel[T](uid, true)
    } else new PostalCodeIdentifierModel[T](uid, false)
  }
}

class PostalCodeIdentifierModel[T <: Text]
(
  override val uid: String,
  val treatAsPostalCode: Boolean
)(implicit tti: TypeTag[T])
  extends UnaryModel[T, PostalCodeMap]("postal code identifier", uid)
    with PostalCodeHelpers {
  def transformFn: Text => PostalCodeMap = input => {
    val rawInput = input.value.getOrElse("")
    val postalCode = findBestPostalCodeMatch(rawInput)
    if (treatAsPostalCode) {
      val (latOption, lngOption) = postalCodeDictionary.getOrElse(postalCode, (None, None))
      (latOption, lngOption) match {
        case (Some(lat), Some(lng)) =>
          PostalCodeMap(Map("postalCode" -> postalCode, "lat" -> lat.toString, "lng" -> lng.toString))
        case _ => PostalCodeMap(Map(postalCode -> "true", "lat" -> "", "lng" -> ""))
      }
    }
    else PostalCodeMap(Map.empty[String, String])
  }
}
