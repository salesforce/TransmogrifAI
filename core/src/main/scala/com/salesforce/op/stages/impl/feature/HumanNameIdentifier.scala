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
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

import scala.io.Source
import scala.util.Try
import scala.reflect.runtime.universe.TypeTag

trait NameCleaner {
  def preProcess(s: String): Array[String] = {
    s.toLowerCase().split("\\s+").map(_.replace("\\P{L}", ""))
  }
}

class HumanNameIdentifier[T <: Text]
(
  uid: String = UID[HumanNameIdentifier[_]],
  operationName: String = "human name identifier"
)
(
  implicit tti: TypeTag[T],
  override val ttiv: TypeTag[T#Value]
) extends UnaryEstimator[T, NameStats](
  uid = uid,
  operationName = operationName
) with NameCleaner {
  // Parameters
  // TODO: Create additional ones for: uniqueness checking, attempting to do name parsing, flag for data source
  val defaultThreshold = new DoubleParam(
    parent = this,
    name = "defaultThreshold",
    doc = "default fraction of entries to be names before treating as name",
    isValid = (value: Double) => {
      ParamValidators.gt(0.0)(value) && ParamValidators.lt(1.0)(value)
    }
  )
  setDefault(defaultThreshold, 0.50)


  def setThreshold(value: Double): this.type = set(defaultThreshold, value)

  // TODO: Extract following code into its own class
  // TODO: Use more robust data sources + start repo for maintaining data
  lazy private val nameDictionary = {
    val nameDictionary = collection.mutable.Set.empty[String]
    val dictionaryPath = "/NameIdentification_JRC.txt"
    val stream = getClass.getResourceAsStream(dictionaryPath)
    val buffer = Source.fromInputStream(stream)
    for {name <- buffer.getLines} {
      nameDictionary += name
    }
    buffer.close
    nameDictionary
  }

  private def extractDouble(dataset: DataFrame): Double = dataset.collect().headOption.getOrElse(Row(0.0)).getDouble(0)

  private def averageBoolCol(dataset: Dataset[Boolean], column: Column): Double = {
    extractDouble(dataset.select(mean(column.cast("integer"))))
  }

  private def guardChecks(dataset: Dataset[T#Value], column: Column): Boolean = {
    val total = dataset.count()
    val numUnique = extractDouble(dataset.select(approx_count_distinct(column).cast(DoubleType)))

    val checks = List(
      // check that in at least 3/4 of the texts there are no more than 10 tokens
      averageBoolCol(dataset.withColumn(
        column.toString, size(split(column, "\\s+")) < 10).asInstanceOf[Dataset[Boolean]], column
      ) > 0.75,
      // check that at least 3/4 of the texts are longer than 3 characters
      averageBoolCol(dataset.withColumn(
        column.toString, length(column) > 3).asInstanceOf[Dataset[Boolean]], column
      ) > 0.75,
      // check that the standard deviation of the text length is greater than a small number
      total < 10 ||
        extractDouble(dataset.select(stddev(length(column)))) > 0.05,
      // check that the number of unique entries is at least 10
      total < 100 || numUnique > 10
    )
    checks.forall(identity)
  }

  // TODO: Eventually, we will want this to perform separate checks in first/last name dictionaries
  // And then map from the original tokens to which dictionaries they were found in
  // which can help us figure out what the order of names is
  private def dictCheck: UserDefinedFunction = udf((s: String) => {
    val tokens = preProcess(s)
    val percentageMatched = tokens.map(token => if (nameDictionary contains token) 1 else 0).sum / tokens.length
    percentageMatched >= 0.5
  }: Boolean)

  private def predictIfName(dataset: Dataset[T#Value], column: Column): Dataset[Boolean] = {
    dataset.select(dictCheck(column).alias(column.toString)).asInstanceOf[Dataset[Boolean]]
  }

  def fitFn(dataset: Dataset[T#Value]): HumanNameIdentifierModel[T] = {
    assert(dataset.schema.fieldNames.length == 1)
    val column = col(dataset.schema.fieldNames.head)
    if (!guardChecks(dataset, column)) {
      new HumanNameIdentifierModel[T](uid, false)
    } else {
      val predictedDF = predictIfName(dataset, column)
      val predictedProb: Double = averageBoolCol(predictedDF, column)
      val treatAsName: Boolean = predictedProb >= $(defaultThreshold)
      new HumanNameIdentifierModel[T](uid, treatAsName)
    }
  }
}


class HumanNameIdentifierModel[T <: Text]
(
  override val uid: String,
  val treatAsName: Boolean
)(implicit tti: TypeTag[T])
  extends UnaryModel[T, NameStats]("human name identifier", uid) with NameCleaner {

  lazy private val genderDictionary = {
    val genderDictionary = collection.mutable.Map.empty[String, Double]
    val dictionaryPath = "/GenderDictionary_SSA.csv"
    val stream = getClass.getResourceAsStream(dictionaryPath)
    val buffer = Source.fromInputStream(stream)
    for {row <- buffer.getLines.drop(1)} {
      val cols = row.split(",").map(_.trim)
      val name = cols(0).toLowerCase().replace("\\P{L}", "")
      val probMale = Try {
        cols(6).toDouble
      }.toOption
      probMale match {
        case Some(prob) => genderDictionary += (name -> prob)
        case None =>
      }
    }
    buffer.close
    genderDictionary
  }

  import NameStats.Keys._
  import NameStats.BooleanStrings._
  import NameStats.GenderStrings._


  def transformFn: Text => NameStats = input => {
    val name = input.value.getOrElse("")
    val tokens = preProcess(name)
    if (treatAsName) {
      val gender = if (tokens.length != 1) GenderNotInferred else {
        genderDictionary.get(tokens.head).map(
          probMale => if (probMale >= 0.5) Male else Female
        ).getOrElse(GenderNA)
      }
      NameStats(Map(
        IsNameIndicator -> True,
        OriginalName -> input.value.getOrElse(""),
        Gender -> gender
      ))
    }
    else NameStats(Map.empty[String, String])
  }
}
