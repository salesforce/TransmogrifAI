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
import com.salesforce.op.utils.stages.NameIdentificationUtils.{
  GenderDictionary,
  NameDictionary,
  tokensToCheckForFirstName
}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.{DoubleParam, ParamValidators}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.MetadataBuilder
import org.apache.spark.sql.{Column, Dataset, SparkSession}
import org.apache.spark.util.SparkUtils.{averageBoolCol, averageDoubleCol, extractDouble}

import scala.reflect.runtime.universe.TypeTag

private[op] trait NameIdentificationFun[T <: Text] extends Logging {
  val spark: SparkSession = SparkSession.builder().getOrCreate()
  import spark.implicits._

  implicit val broadcastNameDict: Broadcast[NameDictionary] = spark.sparkContext.broadcast(NameDictionary())
  implicit val broadcastGenderDict: Broadcast[GenderDictionary] = spark.sparkContext.broadcast(GenderDictionary())

  def preProcess(s: T#Value): Seq[String] = {
    s.getOrElse("").toLowerCase().split("\\s+").map((token: String) =>
      token.replace("\\P{L}", "")
    ).toSeq
  }
  def preProcessUDF: UserDefinedFunction = udf(preProcess _)

  def guardChecks(dataset: Dataset[T#Value], column: Column): Boolean = {
    // TODO: Figure out reasonable values for the timeout
    val total = dataset.rdd.countApprox(timeout = 500).getFinalValue().mean
    val numUnique = extractDouble(dataset.select(approx_count_distinct(column).as[Double]))
    val checks = List(
      // check that in at least 3/4 of the texts there are no more than 10 tokens
      averageBoolCol(dataset.select(
        (size(split(column, "\\s+")) < 10).alias(column.toString).as[Boolean]
      ), column) > 0.75,
      // check that at least 3/4 of the texts are longer than 3 characters
      averageBoolCol(dataset.select(
        (length(column) > 3).alias(column.toString).as[Boolean]
      ), column) > 0.75,
      // check that the standard deviation of the text length is greater than a small number
      total < 10 ||
        extractDouble(dataset.select(stddev(length(column)).as[Double])) > 0.05,
      // check that the number of unique entries is at least 10
      total < 100 || numUnique > 10
    )
    checks.forall(identity)
  }

  def dictCheck(tokens: Seq[String])(implicit dict: Broadcast[NameDictionary]): Double = tokens.map({token: String =>
    if (dict.value.value contains token) 1 else 0
  }).sum.toDouble / tokens.length
  def dictCheckUDF: UserDefinedFunction = {
    udf(dictCheck _)
  }

  def checkForFirstName(tokens: Seq[String])(implicit dict: Broadcast[GenderDictionary]): Seq[Boolean] = {
    tokensToCheckForFirstName.map({i: Int =>
      dict.value.value contains tokens((i + tokens.length) % tokens.length)
    })
  }

  def unaryEstimatorFitFn(
    dataset: Dataset[T#Value], column: Column, threshold: Double
  ): (Double, Boolean, Option[Int]) = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    if (log.isDebugEnabled) dataset.show(truncate = false)
    if (!guardChecks(dataset, column)) {
      (0.0, false, None)
    } else {
      val tokenizedDS = dataset.map(preProcess)
      if (log.isDebugEnabled) tokenizedDS.show(truncate = false)
      // Check if likely to be a name field
      val checkedDS = tokenizedDS.map(dictCheck)
      if (log.isDebugEnabled) checkedDS.show(truncate = false)
      val predictedProb: Double = averageDoubleCol(checkedDS, column)
      logDebug(s"PREDICTED NAME PROB FOR ${column.toString()}: $predictedProb")
      if (predictedProb < threshold) (0.0, false, None)
      else {
        // Also figure out the index of the likely first name
        val checkedForFirstName = tokenizedDS.map(checkForFirstName)
        if (log.isDebugEnabled) checkedForFirstName.show(truncate = false)
        val percentageFirstNameByN = for {i <- tokensToCheckForFirstName} yield {
          // Use one more map to extract the particular boolean result that we need
          val percentageMatched = averageBoolCol(checkedForFirstName.map(
            bools => bools((i + bools.length) % bools.length)
          ), column)
          (percentageMatched, i)
        }
        val (_, bestIndex) = percentageFirstNameByN.maxBy(_._1)
        (predictedProb, true, Some(bestIndex))
      }
    }
  }

  import NameStats.BooleanStrings._
  import NameStats.GenderStrings._
  import NameStats.Keys._

  def identifyGender(tokens: Seq[String], index: Int)(implicit dict: Broadcast[GenderDictionary]): String = {
    val nameToCheckGenderOf = if (tokens.length != 1) {
      // Mod to accept -1 as valid index
      tokens((index + tokens.length) % tokens.length)
    } else tokens.head
    dict.value.value.get(nameToCheckGenderOf).map(
      probMale => if (probMale >= 0.5) Male else Female
    ).getOrElse(GenderNA)
  }

  def transformerFn(treatAsName: Boolean, indexFirstName: Option[Int], input: Text): NameStats = {
    val tokens = preProcess(input.value)
    if (treatAsName) {
      assert(tokens.length == 1 || indexFirstName.isDefined)
      val gender = identifyGender(tokens, indexFirstName.getOrElse(0))
      NameStats(Map(
        IsNameIndicator -> True,
        OriginalName -> input.value.getOrElse(""),
        Gender -> gender
      ))
    }
    else NameStats(Map.empty[String, String])
  }
}


class HumanNameIdentifier[T <: Text]
(
  uid: String = UID[HumanNameIdentifier[T]],
  operationName: String = "human name identifier"
)
(
  implicit tti: TypeTag[T],
  override val ttiv: TypeTag[T#Value]
) extends UnaryEstimator[T, NameStats](
  uid = uid,
  operationName = operationName
) with NameIdentificationFun[T] {
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

  def fitFn(dataset: Dataset[T#Value]): HumanNameIdentifierModel[T] = {
    require(dataset.schema.fieldNames.length == 1, "There is exactly one column in this dataset")
    val column = col(dataset.schema.fieldNames.head)
    val (predictedProb, treatAsName, indexFirstName) = unaryEstimatorFitFn(dataset, column, $(defaultThreshold))

    // modified from: https://docs.transmogrif.ai/en/stable/developer-guide/index.html#metadata
    // get a reference to the current metadata
    val preExistingMetadata = getMetadata()
    // create a new metadataBuilder and seed it with the current metadata
    val metaDataBuilder = new MetadataBuilder().withMetadata(preExistingMetadata)
    // add a new key value pair to the metadata (key is a string and value is a string array)
    metaDataBuilder.putBoolean("treatAsName", treatAsName)
    metaDataBuilder.putLong("predictedNameProb", predictedProb.toLong)
    metaDataBuilder.putLong("indexFirstName", indexFirstName.getOrElse(-1).toLong)
    // TODO: Compute some more stats here
    // package the new metadata, which includes the preExistingMetadata
    // and the updates/additions
    val updatedMetadata = metaDataBuilder.build()
    // save the updatedMetadata to the outputMetadata parameter
    setMetadata(updatedMetadata)

    new HumanNameIdentifierModel[T](uid, treatAsName, indexFirstName = indexFirstName)
  }
}


class HumanNameIdentifierModel[T <: Text]
(
  override val uid: String,
  val treatAsName: Boolean,
  val indexFirstName: Option[Int] = None
)(implicit tti: TypeTag[T])
  extends UnaryModel[T, NameStats]("human name identifier", uid) with NameIdentificationFun[T] {
  // Why doesn't the following line work
  // def transformFn(input: T): NameStats = transformerFn(treatAsName, indexFirstName, input)
  def transformFn: T => NameStats = (input: T) => transformerFn(treatAsName, indexFirstName, input)
}
