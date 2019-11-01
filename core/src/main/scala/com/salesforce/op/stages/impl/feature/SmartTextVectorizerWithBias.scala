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

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Text}
import com.salesforce.op.stages.base.sequence.SequenceModel
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.param.{DoubleParam, ParamValidators}
import org.apache.spark.sql.types.MetadataBuilder
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.reflect.runtime.universe.TypeTag

class SmartTextVectorizerWithBias[T <: Text]
(
  uid: String = UID[SmartTextVectorizerWithBias[T]],
  operationName: String = "smartTxtVecWithBias"
)(implicit tti: TypeTag[T]) extends SmartTextVectorizer(
  uid = uid,
  operationName = operationName
) with NameIdentificationFun[T] {

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

  private var guardCheckResults: Option[Array[Boolean]] = None

  override def fit(dataset: Dataset[_]): SequenceModel[T, OPVector] = {
    // Set instance variable for guardCheck results
    // NOTE: I can also use this trick here to call `unaryEstimatorFitFn` instead here, if that turns out to be faster
    guardCheckResults = Some(inN.map(feature => guardChecks(dataset.asInstanceOf[Dataset[T#Value]], col(feature.name))))
    // then call super
    super.fit(dataset).asInstanceOf[SequenceModel[T, OPVector]]
  }

  override def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val N = dataset.count().toDouble

    val preprocessed: Dataset[Seq[Array[String]]] = dataset.map(_.map(preProcess))
    val checkedInDictionary: Dataset[Seq[Double]] = preprocessed.map(_.map(dictCheck))
    val predictedProbs: Array[Double] = checkedInDictionary.reduce({ (a: Seq[Double], b: Seq[Double]) =>
      // Elementwise sum the contents of each row
      a.zip(b).map { case (x, y) => x + y }
    }).map(_ / N).toArray

    if (log.isDebugEnabled) {
      preprocessed.show(truncate = false)
      checkedInDictionary.show(truncate = false)
      logDebug(predictedProbs.toString)
    }

    val isName: Array[Boolean] = guardCheckResults match {
      case Some(results) => predictedProbs zip results map {
        case (prob, guardCheck) => guardCheck && prob >= $(defaultThreshold)
      }
      case _ => {
        require(false, "Guard check results were not generated but this should not happen.")
        Array.emptyBooleanArray
      }
    }

    val checkedForFirstName: Dataset[Seq[Array[Boolean]]] = preprocessed.map(_.map(checkForFirstName))
    val percentageFirstNameByN: Seq[(Array[Double], Int)] = for {i <- tokensToCheckForFirstName} yield {
      // Use one more map to get the index of the result that we need
      // And then reduce to sum over the rows
      val percentageMatched = checkedForFirstName.map(_.map(
        bools => if (bools((i + bools.length) % bools.length)) 1 else 0
      )).reduce({ (a: Seq[Int], b: Seq[Int]) =>
        // Elementwise sum the contents of each row
        a.zip(b).map { case (x, y) => x + y }
      }).map(_ / N).toArray
      (percentageMatched, i)
    }
    val bestIndexes: Array[Int] = percentageFirstNameByN.foldLeft {
      Array.fill[Int](percentageFirstNameByN.length)(0) zip Array.fill[Double](percentageFirstNameByN.length)(0)
    }{ case (accBestResults, (probs, index)) =>
      accBestResults zip probs map {
        case ((bestIndex, bestProb), newProb) => if (newProb > bestProb) (index, newProb) else (bestIndex, bestProb)
      }
    }.map(_._1)

    // Now, identify the likely gender
    // The values in this new dataset will be (1, 0, 0) for male, (0, 1, 0) for female, and (0, 0, 0) otherwise
    import com.salesforce.op.features.types.NameStats.GenderStrings._
    val inferredGenders: Dataset[Seq[(Int, Int, Int)]] = preprocessed.map((row: Seq[Array[String]]) => {
      row.zip(bestIndexes).map({ case (tokens: Array[String], index: Int) =>
        identifyGender(tokens, index) match {
          case Male => (1, 0, 0)
          case Female => (0, 1, 0)
          case _ => (0, 0, 1)
        }
      })
    })
    val (pctMale, pctFemale, pctOther) = inferredGenders.reduce({ (a, b) =>
      a.zip(b).map({ case ((i, j, k), (x, y, z)) => (i + x, j + y, k + z) })
    }).map({
      case (numMale, numFemale, numOther) => (numMale / N, numFemale / N, numOther / N)
    }).unzip3
    // call SmartTextVectorizer normally
    val modelArgs: SmartTextVectorizerModelArgs = super.fitFn(dataset).asInstanceOf[SmartTextVectorizerModel[T]].args
    val newModelArgs: SmartTextVectorizerModelArgs = modelArgs.copy(isName = isName)

    // modified from: https://docs.transmogrif.ai/en/stable/developer-guide/index.html#metadata
    // get a reference to the current metadata
    val preExistingMetadata = getMetadata()
    // create a new metadataBuilder and seed it with the current metadata
    val metaDataBuilder = new MetadataBuilder().withMetadata(preExistingMetadata)
    // add a new key value pair to the metadata (key is a string and value is a string array)
    metaDataBuilder.putBooleanArray("treatAsName", isName)
    metaDataBuilder.putDoubleArray("predictedNameProb", predictedProbs)
    metaDataBuilder.putDoubleArray("pctMale", pctMale.toArray)
    metaDataBuilder.putDoubleArray("pctFemale", pctFemale.toArray)
    metaDataBuilder.putDoubleArray("pctOther", pctOther.toArray)
    // Also log the above results
    logInfo(s"treatAsName: [${isName.mkString(",")}]")
    logInfo(s"predictedNameProb: [${predictedProbs.mkString(",")}]")
    logInfo(s"pctMale: [${pctMale.mkString(",")}]")
    logInfo(s"pctFemale: [${pctFemale.mkString(",")}]")
    logInfo(s"pctOther: [${pctOther.mkString(",")}]")

    // package the new metadata, which includes the preExistingMetadata
    // and the updates/additions
    val updatedMetadata = metaDataBuilder.build()
    // save the updatedMetadata to the outputMetadata parameter
    setMetadata(updatedMetadata)

    new SmartTextVectorizerModel[T](args = newModelArgs, operationName = operationName, uid = uid)
      .setAutoDetectLanguage(getAutoDetectLanguage)
      .setAutoDetectThreshold(getAutoDetectThreshold)
      .setDefaultLanguage(getDefaultLanguage)
      .setMinTokenLength(getMinTokenLength)
      .setToLowercase(getToLowercase)
      .setTrackTextLen($(trackTextLen))
  }
}
