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

package com.salesforce.op.utils.stages

import com.salesforce.op.features.types.{NameStats, Text}
import com.salesforce.op.utils.json.JsonLike
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, Dataset}
import org.apache.spark.util.SparkUtils.{averageBoolCol, averageDoubleCol, extractDouble}

import scala.io.Source
import scala.util.Try

/**
 * Provides shared helper functions and variables (namely, broadcast dictionaries) for name identification
 * and name to gender transformation.
 * @tparam T     the FeatureType (subtype of Text) to operate over
 */
private[op] trait NameIdentificationFun[T <: Text] extends Logging {
  import com.salesforce.op.utils.stages.NameIdentificationUtils._

  def preProcess(s: T#Value): Seq[String] = {
    s.getOrElse("").toLowerCase().split("\\s+").map((token: String) =>
      token.replace("\\P{L}", "")
    ).toSeq
  }
  def preProcessUDF: UserDefinedFunction = udf(preProcess _)

  def guardChecks(dataset: Dataset[T#Value], column: Column, timeout: Int = 1000): Boolean = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val total = dataset.rdd.countApprox(timeout = timeout).getFinalValue().mean
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

  def dictCheck(tokens: Seq[String], dict: Broadcast[NameDictionary]): Double = {
    tokens.map({ token: String => if (dict.value.value contains token) 1 else 0}).sum.toDouble / tokens.length
  }
  def dictCheckUDF: UserDefinedFunction = {
    udf(dictCheck _)
  }

  def checkForFirstName(tokens: Seq[String], dict: Broadcast[GenderDictionary]): Seq[Boolean] = {
    TokensToCheckForFirstName.map({i: Int =>
      dict.value.value contains tokens((i + tokens.length) % tokens.length)
    })
  }

  def unaryEstimatorFitFn(
    dataset: Dataset[T#Value], column: Column, threshold: Double, timeout: Int = 1000
  ): (Double, Boolean, Option[Int]) = {
    val spark = dataset.sparkSession
    import spark.implicits._
    val broadcastNameDict: Broadcast[NameDictionary] = spark.sparkContext.broadcast(NameDictionary())
    val broadcastGenderDict: Broadcast[GenderDictionary] = spark.sparkContext.broadcast(GenderDictionary())

    if (log.isDebugEnabled) dataset.show(truncate = false)
    if (!guardChecks(dataset, column)) {
      (0.0, false, None)
    } else {
      val tokenizedDS: Dataset[Seq[String]] = dataset.map(preProcess)
      if (log.isDebugEnabled) tokenizedDS.show(truncate = false)
      // Check if likely to be a name field
      val checkedDS: Dataset[Double] = tokenizedDS.map(row => dictCheck(row, broadcastNameDict))
      if (log.isDebugEnabled) checkedDS.show(truncate = false)
      val predictedProb: Double = averageDoubleCol(checkedDS, column)
      logDebug(s"PREDICTED NAME PROB FOR ${column.toString()}: $predictedProb")
      if (predictedProb < threshold) (0.0, false, None)
      else {
        // Also figure out the index of the likely first name
        val checkedForFirstName = tokenizedDS.map(row => checkForFirstName(row, broadcastGenderDict))
        if (log.isDebugEnabled) checkedForFirstName.show(truncate = false)
        val percentageFirstNameByN = for {i <- TokensToCheckForFirstName} yield {
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

  def identifyGender(tokens: Seq[String], index: Int, dict: Broadcast[GenderDictionary]): String = {
    val nameToCheckGenderOf = if (tokens.length != 1) {
      // Mod to accept -1 as valid index
      tokens((index + tokens.length) % tokens.length)
    } else tokens.head
    dict.value.value.get(nameToCheckGenderOf).map(
      probMale => if (probMale >= 0.5) Male else Female
    ).getOrElse(GenderNA)
  }

  def transformerFn(
    treatAsName: Boolean, indexFirstName: Option[Int], input: Text, broadcastGenderDict: Broadcast[GenderDictionary]
  ): NameStats = {
    val tokens = preProcess(input.value)
    if (treatAsName) {
      assert(tokens.length == 1 || indexFirstName.isDefined)
      val gender = identifyGender(tokens, indexFirstName.getOrElse(0), broadcastGenderDict)
      NameStats(Map(
        IsNameIndicator -> True,
        OriginalName -> input.value.getOrElse(""),
        Gender -> gender
      ))
    }
    else NameStats(Map.empty[String, String])
  }
}

/**
 * Defines static values for name identification:
 * - Dictionary filenames and how to read them in
 * - Which parts of a string to check for first name (used in transforming from name to gender)
 *
 * Name and gender data are maintained by and taken from this repository:
 *  https://github.com/MWYang/InternationalNames
 * which itself sources data from:
 *  https://ec.europa.eu/jrc/en/language-technologies/jrc-names
 *  https://github.com/OpenGenderTracking/globalnamedata
 *  https://github.com/first20hours/google-10000-english
 */
object NameIdentificationUtils {
  case class NameDictionary
  (
    value: Set[String] = {
      val nameDictionary = collection.mutable.Set.empty[String]
      val dictionaryPath = "/Names_JRC_Combined.txt"
      val stream = getClass.getResourceAsStream(dictionaryPath)
      val buffer = Source.fromInputStream(stream)
      for {name <- buffer.getLines} {
        nameDictionary += name
      }
      buffer.close
      nameDictionary.toSet[String]
    }
  ) extends JsonLike

  case class GenderDictionary
  (
    value: Map[String, Double] = {
      val genderDictionary = collection.mutable.Map.empty[String, Double]
      val dictionaryPath = "/GenderDictionary_USandUK.csv"
      val stream = getClass.getResourceAsStream(dictionaryPath)
      val buffer = Source.fromInputStream(stream)
      // TODO: Also make use of frequency information in this dictionary
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
      genderDictionary.toMap[String, Double]
    }
  ) extends JsonLike

  // TODO: Eventually, this will be a Seq of case classes to define whether to check an indexed token
  //  (e.g. first or last) or use some RegEx to extract the token to be checked
  val TokensToCheckForFirstName: Seq[Int] = Seq(0, -1)
  val EmptyTokensMap: Map[Int, Int] = Map(TokensToCheckForFirstName map { i => i -> 0 }: _*)
}
