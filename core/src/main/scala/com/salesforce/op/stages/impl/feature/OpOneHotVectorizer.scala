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
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.VectorizerUtils._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.twitter.algebird.Operators._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Converts a sequence of features into a vector keeping the top K most common occurrences of each
 * feature (ie the final vector has length K * number of inputs). Plus an additional column
 * for "other" values - which will capture values that do not make the cut or values not seen
 * in training, and an additional column for empty values unless null tracking is disabled.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid uid for instance
 */
abstract class OpOneHotVectorizer[T <: FeatureType]
(
  operationName: String,
  uid: String = UID[OpOneHotVectorizer[_]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends SequenceEstimator[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults with PivotParams with CleanTextFun with SaveOthersParams
    with TrackNullsParam with MinSupportParam with OneHotFun {

  protected def convertToSeqOfMaps(dataset: Dataset[Seq[T#Value]]): RDD[Seq[Map[String, Int]]]

  protected def makeModel(topValues: Seq[Seq[String]], shouldCleanText: Boolean,
    shouldTrackNulls: Boolean, operationName: String, uid: String): SequenceModel[T, OPVector]

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val shouldCleanText = $(cleanText)
    val shouldTrackNulls = $(trackNulls)

    val rdd: RDD[Seq[Map[String, Int]]] = convertToSeqOfMaps(dataset)

    val countOccurrences: Seq[Map[String, Int]] = {
      if (rdd.isEmpty) Seq.empty[Map[String, Int]]
      else rdd.reduce((a, b) => a.zip(b).map { case (m1, m2) => m1 + m2 })
    }

    // Top K values for each categorical input
    val numToKeep = $(topK)
    val minSup = $(minSupport)
    val topValues: Seq[Seq[String]] =
      countOccurrences.map(m => m.toSeq.filter(_._2 >= minSup).sortBy(v => -v._2 -> v._1).take(numToKeep).map(_._1))

    // build metadata describing output
    val unseen = Option($(unseenName))
    val vecMetadata = makeVectorMetadata(
      shouldTrackNulls = shouldTrackNulls,
      unseen = unseen, topValues = topValues,
      outputName = getOutputFeatureName,
      features = getTransientFeatures(),
      stageName = stageName
    )
    setMetadata(vecMetadata.toMetadata)

    makeModel(
      topValues = topValues,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = shouldTrackNulls,
      operationName = operationName,
      uid = uid
    )
  }
}

abstract class OpOneHotVectorizerModel[T <: FeatureType]
(
  val topValues: Seq[Seq[String]],
  val shouldCleanText: Boolean,
  val shouldTrackNulls: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults with CleanTextFun with OneHotModelFun[T] {

  def transformFn: Seq[T] => OPVector = pivotFn(topValues, shouldCleanText, shouldTrackNulls)

}

/**
 * Converts a sequence of OpSet features into a vector keeping the top K most common occurrences of each
 * feature (ie the final vector has length K * number of inputs). Plus an additional column
 * for "other" values - which will capture values that do not make the cut or values not seen
 * in training, and an additional column for empty values unless null tracking is disabled.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid uid for instance
 */
class OpSetVectorizer[T <: OPSet[_]]
(
  operationName: String = "vecSet",
  uid: String = UID[OpSetVectorizer[_]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends OpOneHotVectorizer[T](operationName = operationName, uid = uid){

  override protected def convertToSeqOfMaps(dataset: Dataset[Seq[T#Value]]): RDD[Seq[Map[String, Int]]] = {
    val shouldCleanText = $(cleanText)

    dataset.rdd.map(_.map(cat =>
      cat.map(v => cleanTextFn(v.toString, shouldCleanText) -> 1).toMap
    ))
  }

  override protected def makeModel(topValues: Seq[Seq[String]], shouldCleanText: Boolean,
    shouldTrackNulls: Boolean, operationName: String, uid: String) =
    new OpSetVectorizerModel(
      topValues = topValues,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = shouldTrackNulls,
      operationName = operationName,
      uid = uid)

}

final class OpSetVectorizerModel[T <: OPSet[_]] private[op]
(
  topValues: Seq[Seq[String]],
  shouldCleanText: Boolean,
  shouldTrackNulls: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OpOneHotVectorizerModel[T](topValues, shouldCleanText, shouldTrackNulls, operationName, uid) {
  override protected def convertToSet(in: T): Set[_] = in.value.toSet
}


/**
 * Converts a sequence of Text features into a vector keeping the top K most common occurrences of each
 * feature (ie the final vector has length K * number of inputs). Plus an additional column
 * for "other" values - which will capture values that do not make the cut or values not seen
 * in training, and an additional column for empty values unless null tracking is disabled.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid uid for instance
 */
class OpTextPivotVectorizer[T <: Text]
(
  operationName: String = "pivotText",
  uid: String = UID[OpSetVectorizer[_]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends OpOneHotVectorizer[T](operationName = operationName, uid = uid){

  override protected def convertToSeqOfMaps(dataset: Dataset[Seq[T#Value]]): RDD[Seq[Map[String, Int]]] = {
    val shouldCleanText = $(cleanText)

    dataset.rdd.map(_.map(cat =>
      cat.map(v => cleanTextFn(v, shouldCleanText) -> 1).toMap
    ))
  }

  override protected def makeModel(topValues: Seq[Seq[String]], shouldCleanText: Boolean,
    shouldTrackNulls: Boolean, operationName: String, uid: String) =
    new OpTextPivotVectorizerModel(
      topValues = topValues,
      shouldCleanText = shouldCleanText,
      shouldTrackNulls = shouldTrackNulls,
      operationName = operationName,
      uid = uid)
}

final class OpTextPivotVectorizerModel[T <: Text] private[op]
(
  topValues: Seq[Seq[String]],
  shouldCleanText: Boolean,
  shouldTrackNulls: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OpOneHotVectorizerModel[T](topValues, shouldCleanText, shouldTrackNulls, operationName, uid) {
  override protected def convertToSet(in: T): Set[_] = in.value.toSet
}

/**
 * One Hot Functionality
 */
private[op] trait OneHotFun {

  protected def makeVectorColumnMetadata(
    shouldTrackNulls: Boolean, unseen: Option[String], topValues: Seq[Seq[String]], features: Array[TransientFeature]
  ): Array[OpVectorColumnMetadata] = {
    for {
      (parentFeature, values) <- features.zip(topValues)
      parentFeatureType = parentFeature.typeName
      // Append other/null indicators for each input (view here to avoid copying the array when appending the string)
      value <-
      if (shouldTrackNulls) values.map(Option(_)).view ++ Array(unseen, Option(TransmogrifierDefaults.NullString))
      else values.map(Option(_)).view :+ unseen
    } yield parentFeature.toColumnMetaData(true).copy(indicatorValue = value)
  }

  protected def makeVectorMetadata(
    shouldTrackNulls: Boolean, unseen: Option[String], topValues: Seq[Seq[String]], outputName: String,
    features: Array[TransientFeature], stageName: String
  ): OpVectorMetadata = {
    val columns = makeVectorColumnMetadata(shouldTrackNulls, unseen, topValues, features)
    OpVectorMetadata(outputName, columns, Transmogrifier.inputFeaturesToHistory(features, stageName))
  }
}

/**
 * One Hot Model Functionality
 */
private[op] trait OneHotModelFun[T <: FeatureType] extends CleanTextFun {
  protected def convertToSet(in: T): Set[_]

  protected def pivotFn(
    topValues: Seq[Seq[String]], shouldCleanText: Boolean, shouldTrackNulls: Boolean
  ): Seq[T] => OPVector = row => {
    // Combine top values for each feature with categorical feature
    val eachPivoted = row.zip(topValues).map { case (cat, top) =>
      val theseCat = convertToSet(cat)
        .groupBy(v => cleanTextFn(v.toString, shouldCleanText))
        .map { case (k, v) => k -> v.size }
      val topPresent = top.zipWithIndex.collect { case (c, i) if theseCat.contains(c) => (i, theseCat(c).toDouble) }
      val notPresent = theseCat.keySet.diff(top.toSet).toSeq
      val notPresentVal = notPresent.map(theseCat).sum.toDouble
      val nullVal = if (theseCat.isEmpty) 1.0 else 0.0
      // Append the other and null entries to the vector (note topPresent is sparse, so use top.length as proxy for K)
      if (shouldTrackNulls) topPresent ++ Array((top.length, notPresentVal), (top.length + 1, nullVal))
      else topPresent :+ (top.length, notPresentVal)
    }

    // Fix indices for sparse vector
    val reindexed = reindex(eachPivoted)
    val vector = makeSparseVector(reindexed)
    vector.toOPVector
  }
}
