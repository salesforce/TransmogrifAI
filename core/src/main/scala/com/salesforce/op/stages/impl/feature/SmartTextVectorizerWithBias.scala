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
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.stages.NameIdentificationUtils.{emptyTokensMap, tokensToCheckForFirstName}
import com.twitter.algebird.Operators._
import org.apache.spark.ml.param.{DoubleParam, ParamValidators}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.MetadataBuilder

import scala.reflect.runtime.universe.TypeTag

case class NameIdentificationResults
(
  predictedNameProb: Double = 0.0,
  tokenInFirstNameDictionary: Map[Int, Int] = emptyTokensMap,
  tokenIsMale: Map[Int, Int] = emptyTokensMap,
  tokenIsFemale: Map[Int, Int] = emptyTokensMap,
  tokenIsOther: Map[Int, Int] = emptyTokensMap
) extends JsonLike

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

  var guardCheckResults: Option[Array[Boolean]] = None

  override def fit(dataset: Dataset[_]): SequenceModel[T, OPVector] = {
    // Set instance variable for guardCheck results
    guardCheckResults = Some(inN.map(feature => guardChecks(dataset.asInstanceOf[Dataset[T#Value]], col(feature.name))))
    // then call super
    super.fit(dataset).asInstanceOf[SequenceModel[T, OPVector]]
  }

  override def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    def aggregateTwoResults(
      one: NameIdentificationResults, two: NameIdentificationResults
    ): NameIdentificationResults = {
      NameIdentificationResults(
        one.predictedNameProb + two.predictedNameProb,
        one.tokenInFirstNameDictionary + two.tokenInFirstNameDictionary,
        one.tokenIsMale + two.tokenIsMale,
        one.tokenIsFemale + two.tokenIsFemale,
        one.tokenIsOther + two.tokenIsOther
      )
    }

    def aggregateSeqResults(
      one: Seq[NameIdentificationResults], two: Seq[NameIdentificationResults]
    ): Seq[NameIdentificationResults] = {
      one zip two map { case (x, y) => aggregateTwoResults(x, y) }
    }

    import com.salesforce.op.features.types.NameStats.GenderStrings._
    def computeResults(input: T#Value): NameIdentificationResults = {
      val tokens: Seq[String] = preProcess(input)
      val (firstHalf, secondHalf) = (tokensToCheckForFirstName map { index: Int =>
        val (inFirstNameDict, isMale, isFemale, isOther) = identifyGender(tokens, index) match {
          case Male => (1, 1, 0, 0)
          case Female => (1, 0, 1, 0)
          case _ => (0, 0, 0, 1)
        }
        ((index -> inFirstNameDict, index -> isMale), (index -> isFemale, index -> isOther))
      }).unzip
      val (inFirstNameDictSeq, isMaleSeq) = firstHalf.unzip
      val (isFemaleSeq, isOtherSeq) = secondHalf.unzip
      NameIdentificationResults(
        dictCheck(tokens),
        Map(inFirstNameDictSeq: _*),
        Map(isMaleSeq: _*),
        Map(isFemaleSeq: _*),
        Map(isOtherSeq: _*)
      )
    }

    val rdd = dataset.rdd
    // TODO: Figure out reasonable values for the timeout
    val N = rdd.countApprox(timeout = 500).getFinalValue().mean
    val zeroValue = Seq.fill[NameIdentificationResults](inN.length)(NameIdentificationResults())
    val agg = rdd.treeAggregate[Seq[NameIdentificationResults]](zeroValue)(
      combOp = aggregateSeqResults, seqOp = {
        case (result, row) => aggregateSeqResults(result, row.map(computeResults))
      }
    )

    val predictedProbs = agg map { _.predictedNameProb / N }
    // TODO: Move guard check to aggregation?
    // Transform the guard check into two collections: a collection of transforms and a collection of conditions
    val isName = guardCheckResults match {
      case Some(results) => predictedProbs zip results map {
        case (prob, guardCheck) => guardCheck && prob >= $(defaultThreshold)
      }
      case _ => {
        throw new RuntimeException("Guard check results were not generated but this should not happen.")
      }
    }
    val bestIndexes: Seq[Int] = agg map { result: NameIdentificationResults =>
      val (bestIndex, _) = if (result.tokenInFirstNameDictionary.isEmpty) (0, 0)
      else result.tokenInFirstNameDictionary.maxBy(_._2)
      bestIndex
    }
    val (pctMale, pctFemale, pctOther) = (agg zip bestIndexes).map {
      case (result: NameIdentificationResults, index: Int) =>
      (
        result.tokenIsMale.getOrElse(index, 0),
        result.tokenIsFemale.getOrElse(index, 0),
        result.tokenIsOther.getOrElse(index, 0)
      )
    }.map{ case (numMale: Int, numFemale: Int, numOther: Int) => (numMale / N, numFemale / N, numOther / N) }.unzip3

    // call SmartTextVectorizer normally
    val modelArgs: SmartTextVectorizerModelArgs = super.fitFn(dataset).asInstanceOf[SmartTextVectorizerModel[T]].args
    val newModelArgs: SmartTextVectorizerModelArgs = modelArgs.copy(isName = isName.toArray)

    // modified from: https://docs.transmogrif.ai/en/stable/developer-guide/index.html#metadata
    // get a reference to the current metadata
    val preExistingMetadata = getMetadata()
    // create a new metadataBuilder and seed it with the current metadata
    val metaDataBuilder = new MetadataBuilder().withMetadata(preExistingMetadata)
    // add a new key value pair to the metadata (key is a string and value is a string array)
    metaDataBuilder.putBooleanArray("treatAsName", isName.toArray)
    metaDataBuilder.putDoubleArray("predictedNameProb", predictedProbs.toArray)
    metaDataBuilder.putDoubleArray("bestIndexes", bestIndexes.map(_.toDouble).toArray)
    metaDataBuilder.putDoubleArray("pctMale", pctMale.toArray)
    metaDataBuilder.putDoubleArray("pctFemale", pctFemale.toArray)
    metaDataBuilder.putDoubleArray("pctOther", pctOther.toArray)
    // Also log the above results
    logInfo(s"treatAsName: [${isName.mkString(",")}]")
    logInfo(s"predictedNameProb: [${predictedProbs.mkString(",")}]")
    logInfo(s"bestIndexes: [${bestIndexes.mkString(",")}]")
    logInfo(s"pctMale: [${pctMale.mkString(",")}]")
    logInfo(s"pctFemale: [${pctFemale.mkString(",")}]")
    logInfo(s"pctOther: [${pctOther.mkString(",")}]")
    // Get a small sample of rows to log for sanity checking that my code did identify names
    val numSamples = if (N > 100) 5 else 1
    for { (row, index) <- dataset.sample(fraction = numSamples / N).collect() zip (1 to numSamples)} {
      val sample = row.map(_.getOrElse(" ")).toArray
      metaDataBuilder.putStringArray(s"Sample #$index", sample)
      logInfo(s"Sample #$index: ${sample.mkString(" |;| ")}")
    }
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
