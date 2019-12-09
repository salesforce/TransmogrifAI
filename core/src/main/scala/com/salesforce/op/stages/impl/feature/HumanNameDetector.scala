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
import com.salesforce.op.utils.stages.NameDetectUtils.{GenderDictionary, NameDictionary}
import com.salesforce.op.utils.stages.{GenderDetectStrategy, NameDetectFun, NameDetectStats, NameDetectUtils}
import com.twitter.algebird.Operators._
import com.twitter.algebird.{HyperLogLogMonoid, Semigroup}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{DoubleParam, ParamValidators}
import org.apache.spark.sql.types.MetadataBuilder
import org.apache.spark.sql._

import scala.reflect.runtime.universe.TypeTag

/**
 * Unary estimator for identifying whether a single Text column is a name or not. If the column does appear to be a
 * name, a custom map will be returned that contains the guessed gender for each entry. If the column does not appear
 * to be a name, then the output will be an empty map.
 * @param uid           uid for instance
 * @param operationName unique name of the operation this stage performs
 * @param tti           type tag for input
 * @param ttiv          type tag for input value
 * @tparam T            the FeatureType (subtype of Text) to operate over
 */
class HumanNameDetector[T <: Text]
(
  uid: String = UID[HumanNameDetector[T]],
  operationName: String = "human name detector"
)
(
  implicit tti: TypeTag[T],
  override val ttiv: TypeTag[T#Value]
) extends UnaryEstimator[T, NameStats](
  uid = uid,
  operationName = operationName
) with NameDetectFun[T] {

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

  def fitFn(dataset: Dataset[T#Value]): HumanNameDetectorModel[T] = {
    val spark = dataset.sparkSession
    // Load name and gender data into broadcast variables
    val broadcastNameDict: Broadcast[NameDictionary] = spark.sparkContext.broadcast(NameDictionary())
    val broadcastGenderDict: Broadcast[GenderDictionary] = spark.sparkContext.broadcast(GenderDictionary())
    // Create HyperLogLog factory
    val hllMonoid = new HyperLogLogMonoid(NameDetectUtils.HLLBits)
    // Create Spark encoder for our accumulator class
    implicit val nameDetectStatsEncoder: Encoder[NameDetectStats] = Encoders.kryo[NameDetectStats]
    // Tell Algebird that our accumulator class is also a monoid
    implicit val nameDetectStatsMonoid: Semigroup[NameDetectStats] = NameDetectStats.monoid

    val aggResults: NameDetectStats = dataset.map(
      computeResults(_, broadcastNameDict, broadcastGenderDict, hllMonoid)
    ).reduce(_ + _)

    val guardChecksPassed = performGuardChecks(aggResults.guardCheckQuantities, hllMonoid)
    val predictedNameProb = aggResults.dictCheckResult.value
    // TODO: Delete this
    require(
      0.0 <= predictedNameProb && predictedNameProb <= predictedNameProb,
      f"Predicted name probability must be in [0, 1]: $predictedNameProb"
    )
    val treatAsName = guardChecksPassed && predictedNameProb >= $(defaultThreshold)
    val (bestStrategy, _) = aggResults.genderResultsByStrategy.minBy(_._2.numOther)
    val genderDetectStrategy = if (treatAsName) Some(GenderDetectStrategy.fromString(bestStrategy)) else None

    // TODO: Delete these debug logs
    import spark.implicits._
    dataset.map(preProcess).show(truncate = false)
    dataset.map(s => dictCheck(preProcess(s), broadcastNameDict)).show(truncate = false)
    println(aggResults)
    println(guardChecksPassed)
    println(predictedNameProb)
    println(genderDetectStrategy)

    // modified from: https://docs.transmogrif.ai/en/stable/developer-guide/index.html#metadata
    val preExistingMetadata = getMetadata()
    val metaDataBuilder = new MetadataBuilder().withMetadata(preExistingMetadata)
    metaDataBuilder.putBoolean("treatAsName", treatAsName)
    metaDataBuilder.putLong("predictedNameProb", predictedNameProb.toLong)
    metaDataBuilder.putString("bestStrategy", bestStrategy)
    val updatedMetadata = metaDataBuilder.build()
    setMetadata(updatedMetadata)

    new HumanNameDetectorModel[T](uid, treatAsName, genderDetectStrategy)
  }
}


class HumanNameDetectorModel[T <: Text]
(
  override val uid: String,
  val treatAsName: Boolean,
  val genderDetectStrategy: Option[GenderDetectStrategy] = None
)(implicit tti: TypeTag[T])
  extends UnaryModel[T, NameStats]("human name detector", uid) with NameDetectFun[T] {

  var broadcastGenderDict: Option[Broadcast[GenderDictionary]] = None

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark: SparkSession = dataset.sparkSession
    this.broadcastGenderDict = Some(spark.sparkContext.broadcast(GenderDictionary()))
    super.transform(dataset)
  }

  import NameStats.BooleanStrings._
  import NameStats.Keys._
  def transformFn: T => NameStats = (input: T) => {
    val tokens = preProcess(input.value)
    if (treatAsName) {
      require(genderDetectStrategy.nonEmpty, "There must be a gender extraction strategy if treating as name.")
      require(this.broadcastGenderDict.nonEmpty, "Gender dictionary broadcast variable was not initialized correctly.")
      val gender: String = identifyGender(input.value, tokens, genderDetectStrategy.get, this.broadcastGenderDict.get)
      NameStats(Map(
        IsNameIndicator -> True,
        OriginalName -> input.value.getOrElse(""),
        Gender -> gender
      ))
    }
    else NameStats(Map.empty[String, String])
  }
}
