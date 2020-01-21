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

import com.fasterxml.jackson.core.JsonGenerator
import com.fasterxml.jackson.databind.SerializerProvider
import com.fasterxml.jackson.databind.ser.std.StdSerializer
import com.salesforce.op.features.types.NameStats.GenderValue
import com.salesforce.op.features.types.NameStats.GenderValue._
import com.salesforce.op.features.types.Text
import com.salesforce.op.stages.impl.feature.{GenderDetectStrategy, TextTokenizer}
import com.salesforce.op.utils.json.{JsonLike, JsonUtils, SerDes}
import com.twitter.algebird._
import com.twitter.algebird.macros.caseclass
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam, ParamValidators, Params}
import org.apache.spark.sql.{Encoder, Encoders, SparkSession}

import scala.io.Source
import scala.util.Try
import scala.util.matching.Regex

/**
 * Provides shared helper functions and variables for name identification and name to gender transformation.
 */
private[op] trait NameDetectFun[T <: Text] extends Logging with NameDetectParams {
  import GenderDetectStrategy._
  import NameDetectUtils._

  private[op] def preProcess(input: Text): Seq[String] = {
    TextTokenizer.tokenize(input).tokens.toArray
  }

  private[op] def computeGuardCheckQuantities(
    input: T#Value,
    tokens: Seq[String],
    hllMonoid: HyperLogLogMonoid
  ): GuardCheckStats = {
    input match {
      case None => GuardCheckStats(0, 0, MomentsGroup.zero, hllMonoid.zero)
      case Some(text) =>
        val textLength = text.length
        GuardCheckStats(
          countBelowMaxNumTokens = if (tokens.length < $(guardMaxNumberOfTokens)) 1 else 0,
          countAboveMinCharLength = if (textLength >= $(guardMinTextLength)) 1 else 0,
          approxMomentsOfTextLength = Moments(textLength),
          approxNumUnique = hllMonoid.create(text.getBytes)
        )
    }
  }

  private[op] def performGuardChecks(stats: GuardCheckStats, hllMonoid: HyperLogLogMonoid): Boolean = {
    val N: Double = stats.approxMomentsOfTextLength.count.toDouble
    val checks = List(
      // check that in at least 3/4 of the texts there are no more than 10 tokens
      (stats.countBelowMaxNumTokens / N) > $(guardPctMaxNumberOfTokens),
      // check that at least 3/4 of the texts are longer than 3 characters
      (stats.countAboveMinCharLength / N) > $(guardPctMinTextLength),
      // check that the standard deviation of the text length is greater than a small number
      N < $(guardMinCountForStdDevCheck) ||
        stats.approxMomentsOfTextLength.stddev > $(guardMinStdDev),
      // check that the number of unique entries is at least 10
      N < $(guardMinCountForUniqueCheck) ||
        hllMonoid.sizeOf(stats.approxNumUnique).estimate >= $(guardMinUniqueCheck)
    )
    checks.forall(identity)
  }

  private[op] def dictCheck(tokens: Seq[String], dict: NameDictionary): Double = {
    if (tokens.isEmpty) 0.0 else {
      tokens.map({ token: String => if (dict contains token) 1.0 else 0.0}).sum / tokens.length
    }
  }

  private[op] def findHonorific(tokens: Seq[String]): GenderValue = {
    tokens collect {
      case v if MaleHonorifics.contains(v) => Male
      case v if FemaleHonorifics.contains(v) => Female
    } match {
      case Seq(elem) => elem
      case _ => GenderNA // Both no matches and more than one match should be NA
    }
  }

  private[op] def genderDictCheck(nameToCheckGenderOf: T#Value, genderDict: GenderDictionary): GenderValue = {
    nameToCheckGenderOf.flatMap(genderDict.get).map(
      probMale => if (probMale >= 0.5) Male else Female
    ).getOrElse(GenderNA)
  }

  private[op] def identifyGender(
    input: T#Value,
    tokens: Seq[String],
    strategy: GenderDetectStrategy,
    genderDict: GenderDictionary
  ): GenderValue = {
    input match {
      case None => GenderNA
      case Some(text) =>
        strategy match {
          case FindHonorific() => findHonorific(tokens)
          case ByIndex(index) =>
            val nameToCheckGenderOf = tokens.lift(index)
            genderDictCheck(nameToCheckGenderOf, genderDict)
          case ByLast() =>
            val nameToCheckGenderOf = tokens.lastOption
            genderDictCheck(nameToCheckGenderOf, genderDict)
          case ByRegex(pattern) =>
            text match {
              case pattern(matchedGroup) =>
                val nameToCheckGenderOf = preProcess(Text(matchedGroup)).headOption
                genderDictCheck(nameToCheckGenderOf, genderDict)
              case _ => GenderNA
            }
          case _ =>
            logError("Unimplemented gender detection strategy found")
            GenderNA
        }
    }
  }

  private[op] def computeGenderResultsByStrategy(
    input: T#Value,
    tokens: Seq[String],
    genderDict: GenderDictionary
  ): Map[String, GenderStats] = {
    GenderDetectStrategies map { strategy: GenderDetectStrategy =>
      val genderResult: GenderValue = identifyGender(input, tokens, strategy, genderDict)
      implicit def booleanToInt(v: Boolean): Int = if (v) 1 else 0
      strategy.toString -> GenderStats(genderResult == Male, genderResult == Female, genderResult == GenderNA)
    } toMap
  }

  private[op] def computeResults(
    input: T#Value,
    nameDict: Broadcast[NameDictionary],
    genderDict: Broadcast[GenderDictionary],
    hll: HyperLogLogMonoid
  ): NameDetectStats = input match {
    case None if $(ignoreNulls) => NameDetectStats.empty
    case _ =>
      val tokens = preProcess(Text(input))
      NameDetectStats(
        computeGuardCheckQuantities(input, tokens, hll),
        AveragedValue(1L, dictCheck(tokens, nameDict.value)),
        computeGenderResultsByStrategy(input, tokens, genderDict.value)
      )
  }

  private[op] def makeMapFunction(spark: SparkSession): T#Value => NameDetectStats = {
    val broadcastNameDict: Broadcast[NameDictionary] = spark.sparkContext.broadcast(DefaultNameDictionary)
    val broadcastGenderDict: Broadcast[GenderDictionary] = spark.sparkContext.broadcast(DefaultGenderDictionary)
    val hllMonoid = new HyperLogLogMonoid(NameDetectUtils.HLLBits)

    computeResults(_, broadcastNameDict, broadcastGenderDict, hllMonoid)
  }

  private[op] def computeTreatAsName(results: NameDetectStats): Boolean = {
    val hllMonoid = new HyperLogLogMonoid(NameDetectUtils.HLLBits)
    val guardChecksPassed = performGuardChecks(results.guardCheckQuantities, hllMonoid)
    val predictedNameProb = results.dictCheckResult.value
    guardChecksPassed && predictedNameProb >= $(nameThreshold)
  }

  private[op] def orderGenderStrategies(results: NameDetectStats): Seq[GenderDetectStrategy] = {
    val ordered: Seq[(String, GenderStats)] = results.genderResultsByStrategy.toSeq.sortBy(_._2.numOther)
    ordered map { case (strategy, _) => GenderDetectStrategy.fromString(strategy) }
  }
}

/**
 * Defines static values for name identification:
 * - Dictionary filenames and how to read them in
 * - The number of bits to use for HyperLogLog unique entry detection (part of guard checks)
 * - List of male/female honorifics (English only so far)
 * - The different gender detection strategies to try
 *
 * Name and gender data are maintained by and taken from this repository:
 *  https://github.com/MWYang/InternationalNames
 * which itself sources data from:
 *  https://ec.europa.eu/jrc/en/language-technologies/jrc-names
 *  https://github.com/OpenGenderTracking/globalnamedata
 *  https://github.com/first20hours/google-10000-english
 */
private[op] object NameDetectUtils {
  import GenderDetectStrategy._

  type NameDictionary = Set[String]
  lazy val DefaultNameDictionary: NameDictionary = {
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
  // Use the following line to use the smaller but less noisy gender dictionary as a source for names
  // val DefaultNameDictionary: NameDictionary = DefaultGenderDictionary.value.keySet

  type GenderDictionary = Map[String, Double]
  lazy val DefaultGenderDictionary: GenderDictionary = {
    val genderDictionary = collection.mutable.Map.empty[String, Double]
    val dictionaryPath = "/GenderDictionary_USandUK.csv"
    val stream = getClass.getResourceAsStream(dictionaryPath)
    val buffer = Source.fromInputStream(stream)
    // In the future, we could also make use of frequency information in this dictionary
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

  /**
   * Number of bits used for hashing in HyperLogLog (HLL). Error is about 1.04/sqrt(2^{bits}).
   * Default is 12 bits for 1% error which means each HLL instance is about 2^{12} = 4kb per instance.
   */
  val HLLBits = 12

  val MaleHonorifics: Set[String] = Set("mr", "mister", "sir")
  val FemaleHonorifics: Set[String] = Set("ms", "mrs", "miss", "madam")
  val AllHonorifics: Set[String] = MaleHonorifics ++ FemaleHonorifics

  /**
   * The strategies to use for transforming name to gender; Order does not matter.
   *
   * The first RegEx pattern will extract all text after the first comma;
   * The second RegEx pattern will extract all text after both the first comma and the immediately next token,
   *   which accounts for patterns like `LastName, Honorific FirstName MiddleNames`
   */
  val TextAfterFirstComma: Regex = """.*,(.*)""".r
  val TextAfterFirstCommaAndNextToken: Regex = """.*,\s+.*?\s+(.*)""".r
  val GenderDetectStrategies: Seq[GenderDetectStrategy] = Seq(
    FindHonorific(), ByIndex(0), ByLast(), ByRegex(TextAfterFirstComma), ByRegex(TextAfterFirstCommaAndNextToken)
  )
}

private[op] case class GuardCheckStats
(
  countBelowMaxNumTokens: Int = 0,
  countAboveMinCharLength: Int = 0,
  approxMomentsOfTextLength: Moments = MomentsGroup.zero,
  approxNumUnique: HLL = new HyperLogLogMonoid(NameDetectUtils.HLLBits).zero
)

private[op] case class GenderStats(numMale: Int = 0, numFemale: Int = 0, numOther: Int = 0)

/**
 * Defines the case class monoid that will accumulate stats on name detection in a single pass over the data
 * @param guardCheckQuantities     a GuardCheckStats object that uses Algebird approximate algorithms to compute stats
 * @param dictCheckResult          an Algebird AveragedValue object to automatically compute the percentage of name
 *                                 tokens per entry, averaged over all rows
 * @param genderResultsByStrategy  a map from the serialized GenderDetectStrategy to GenderStats case class;
 *                                 the `numOther` value will be used to sort and find the best gender detection strategy
 */
private[op] case class NameDetectStats
(
  guardCheckQuantities: GuardCheckStats,
  dictCheckResult: AveragedValue,
  genderResultsByStrategy: Map[String, GenderStats]
) extends JsonLike {
  override def toJson(pretty: Boolean): String = {
    val hllMonoid = new HyperLogLogMonoid(NameDetectUtils.HLLBits)
    val serializer = new StdSerializer[HLL](classOf[HLL]) {
      override def serialize(value: HLL, gen: JsonGenerator, provider: SerializerProvider): Unit = {
        val sizeEstimate = hllMonoid.sizeOf(value)
        gen.writeArray(
          Array(sizeEstimate.min, sizeEstimate.estimate, sizeEstimate.max, sizeEstimate.probWithinBounds),
          0, 4
        )
      }
    }
    val deserializer = null // will not need to deserialize HLL values
    JsonUtils.toJsonString(this, pretty = pretty, Seq(SerDes[HLL](classOf[HLL], serializer, deserializer)))
  }
}
private[op] case object NameDetectStats {
  def monoid: Monoid[NameDetectStats] = {
    implicit val hllMonoid: Monoid[HLL] = new HyperLogLogMonoid(NameDetectUtils.HLLBits)
    implicit val momentsMonoid: Monoid[Moments] = MomentsAggregator.monoid
    implicit val averagedValueMonoid: Monoid[AveragedValue] = AveragedGroup
    implicit val guardCheckStatsMonoid: Monoid[GuardCheckStats] = caseclass.monoid[GuardCheckStats]
    implicit val genderStatsMonoid: Monoid[GenderStats] = caseclass.monoid[GenderStats]
    caseclass.monoid[NameDetectStats]
  }

  def empty: NameDetectStats = NameDetectStats(GuardCheckStats(), AveragedGroup.zero, Map.empty[String, GenderStats])

  val kryo: Encoder[NameDetectStats] = Encoders.kryo[NameDetectStats]
}

private[op] trait NameDetectParams extends Params {
  val nameThreshold = new DoubleParam(
    parent = this,
    name = "nameThreshold",
    doc = "fraction of entries to be names before treating as name",
    isValid = (value: Double) => {
      ParamValidators.gt(0.0)(value) && ParamValidators.lt(1.0)(value)
    }
  )
  setDefault(nameThreshold, 0.50)
  def setThreshold(value: Double): this.type = set(nameThreshold, value)

  val ignoreNulls = new BooleanParam(
    parent = this,
    name = "ignoreNulls",
    doc = "whether to ignore null values when detecting names and gender"
  )
  setDefault(ignoreNulls, true)
  def setIgnoreNulls(value: Boolean): this.type = set(ignoreNulls, value)

  val guardMaxNumberOfTokens = new IntParam(
    parent = this,
    name = "guardMaxNumberOfTokens",
    doc =
      "maximum (exclusive) number of tokens per entry before not treating as name (helps exclude sentences/paragraphs)",
    isValid = (value: Int) => ParamValidators.gt(0)(value)
  )
  setDefault(guardMaxNumberOfTokens, 10)
  val guardPctMaxNumberOfTokens = new DoubleParam(
    parent = this,
    name = "guardPctMaxNumberOfTokens",
    doc = "fraction of entries to have less than `guardMaxNumberOfTokens` in order to possibly treat as name",
    isValid = (value: Double) => {
      ParamValidators.gt(0.0)(value) && ParamValidators.lt(1.0)(value)
    }
  )
  setDefault(guardPctMaxNumberOfTokens, 0.75)

  val guardMinTextLength = new IntParam(
    parent = this,
    name = "guardMinTextLength",
    doc = "minimum (inclusive) length per entry before not treating as name (helps exclude very short text)",
    isValid = (value: Int) => ParamValidators.gtEq(0)(value)
  )
  setDefault(guardMinTextLength, 3)
  val guardPctMinTextLength = new DoubleParam(
    parent = this,
    name = "guardPctMinTextLength",
    doc = "fraction of entries to have more than `guardMinTextLength` in order to possibly treat as name",
    isValid = (value: Double) => {
      ParamValidators.gt(0.0)(value) && ParamValidators.lt(1.0)(value)
    }
  )
  setDefault(guardPctMinTextLength, 0.75)

  val guardMinCountForStdDevCheck = new IntParam(
    parent = this,
    name = "guardMinCountForStdDevCheck",
    doc = "minimum (inclusive) number of entries before running standard deviation check",
    isValid = (value: Int) => ParamValidators.gtEq(0)(value)
  )
  setDefault(guardMinCountForStdDevCheck, 10)
  val guardMinStdDev = new DoubleParam(
    parent = this,
    name = "guardMinStdDev",
    doc = "minimum (exclusive) standard deviation in order to possibly treat as name",
    isValid = (value: Double) => ParamValidators.gt(0.0)(value)
  )
  setDefault(guardMinStdDev, 0.05)

  val guardMinCountForUniqueCheck = new IntParam(
    parent = this,
    name = "guardMinCountForUniqueCheck",
    doc = "minimum (inclusive) number of entries before running uniqueness check",
    isValid = (value: Int) => ParamValidators.gtEq(0)(value)
  )
  setDefault(guardMinCountForUniqueCheck, 10)
  val guardMinUniqueCheck = new IntParam(
    parent = this,
    name = "guardMinUniqueCheck",
    doc = "minimum (inclusive) number of unique entries in order to possibly treat as name",
    isValid = (value: Int) => ParamValidators.gtEq(0)(value)
  )
  setDefault(guardMinUniqueCheck, 10)

  def setGuardCheckValues(
    maxNumberOfTokens: Int = $(guardMaxNumberOfTokens),
    pctMaxNumberOfTokens: Double = $(guardPctMaxNumberOfTokens),
    minTextLength: Int = $(guardMinTextLength),
    pctMinTextLength: Double = $(guardPctMinTextLength),
    minCountForStdDevCheck: Int = $(guardMinCountForStdDevCheck),
    minStdDev: Double = $(guardMinStdDev),
    minCountForUniqueCheck: Int = $(guardMinCountForUniqueCheck),
    minUniqueCheck: Int = $(guardMinUniqueCheck)
  ): this.type = {
    set(guardMaxNumberOfTokens, maxNumberOfTokens)
    set(guardPctMaxNumberOfTokens, pctMaxNumberOfTokens)
    set(guardMinTextLength, minTextLength)
    set(guardPctMinTextLength, pctMinTextLength)
    set(guardMinCountForStdDevCheck, minCountForStdDevCheck)
    set(guardMinStdDev, minStdDev)
    set(guardMinCountForUniqueCheck, minCountForUniqueCheck)
    set(guardMinUniqueCheck, minUniqueCheck)
  }
}
