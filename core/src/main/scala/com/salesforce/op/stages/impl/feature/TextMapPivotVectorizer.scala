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
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.stages.impl.feature.VectorizerUtils._
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.Dataset

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

/**
 * Converts a sequence of KeyString features into a vector keeping the top K most common occurrences of each
 * key in the maps for that feature (ie the final vector has length k * number of keys * number of features).
 * Each key found will also generate an other column which will capture values that do not make the cut or where not
 * seen in training. Note that any keys not seen in training will be ignored.
 *
 * @param uid uid for instance
 */
class TextMapPivotVectorizer[T <: OPMap[String]]
(
  uid: String = UID[TextMapPivotVectorizer[T]]
)(implicit tti: TypeTag[T])
  extends SequenceEstimator[T, OPVector](operationName = "vecPivotTextMap", uid = uid)
    with VectorizerDefaults with PivotParams with MapPivotParams with TextParams
    with MapStringPivotHelper with CleanTextMapFun with MinSupportParam with TrackNullsParam
    with MaxPercentageCardinalityParams with CountUniqueMapFun {


  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = $(cleanText)

    def convertToMapOfMaps(mapIn: Map[String, String]): MapMap = mapIn.map { case (k, v) => k -> Map(v -> 1L) }
    implicit val classTag: ClassTag[T#Value] = ReflectionUtils.classTagForWeakTypeTag[T#Value]
    implicit val spark = dataset.sparkSession
    implicit val kryo = new KryoSerializer(spark.sparkContext.getConf)
    val (uniqueCounts, n) = countMapUniques(dataset, size = inN.length, bits = $(bits))

    val percentFilter = uniqueCounts.flatMap(_.map{ case (k, v) =>
      k -> (v.estimatedSize / n < $(maxPercentageCardinality))}.toSeq).toMap
    val filteredDataset = filterHighCardinality(dataset, percentFilter)

    val categoryMaps: Dataset[SeqMapMap] =
      getCategoryMaps(filteredDataset, convertToMapOfMaps, shouldCleanKeys, shouldCleanValues)


    val topValues: SeqSeqTupArr = getTopValues(categoryMaps, inN.length, $(topK), $(minSupport))


    val vectorMeta = makeOutputVectorMetadata(topValues, inN, operationName, getOutputFeatureName,
      stageName, $(trackNulls))
    setMetadata(vectorMeta.toMetadata)

    new TextMapPivotVectorizerModel[T](
      topValues = topValues,
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues,
      trackNulls = $(trackNulls),
      operationName = operationName,
      uid = uid
    )
  }
}


final class TextMapPivotVectorizerModel[T <: OPMap[String]] private[op]
(
  val topValues: Seq[Seq[(String, Array[String])]],
  val shouldCleanKeys: Boolean,
  val shouldCleanValues: Boolean,
  val trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults with TextMapPivotVectorizerModelFun[T] {

  def transformFn: Seq[T] => OPVector = pivotFn(
    topValues = topValues,
    shouldCleanKeys = shouldCleanKeys,
    shouldCleanValues = shouldCleanValues,
    shouldTrackNulls = trackNulls
  )
}

/**
 * Text Map Pivot Vectorizer Model Functionality
 */
private[op] trait TextMapPivotVectorizerModelFun[T <: OPMap[String]] extends CleanTextMapFun {
  protected def pivotFn
  (
    topValues: Seq[Seq[(String, Array[String])]], shouldCleanKeys: Boolean, shouldCleanValues: Boolean,
    shouldTrackNulls: Boolean
  ): Seq[T] => OPVector = row => {
    // Combine top values for each feature with map feature
    val eachPivoted =
      row.zip(topValues).map { case (map, topMap) =>
        val cleanedMap = cleanMap(map.value, shouldCleanKeys, shouldCleanValues)

        topMap.map { case (mapKey, top) =>
          val sizeOfVector = top.length
          cleanedMap.get(mapKey) match {
            case None => if (shouldTrackNulls) Seq(sizeOfVector + 1 -> 1.0) else Seq(sizeOfVector -> 0.0)
            case Some(cv) =>
              val v = top.indexOf(cv) match {
                case i if i < 0 => Seq(sizeOfVector -> 1.0)
                case i => Seq(i -> 1.0, sizeOfVector -> 0.0)
              }
              if (shouldTrackNulls) v ++ Seq(sizeOfVector + 1 -> 0.0) else v
          }
        }
      }
    // Fix indices for sparse vector
    val reindexed = reindex(eachPivoted.map(reindex))
    makeSparseVector(reindexed).toOPVector
  }
}
