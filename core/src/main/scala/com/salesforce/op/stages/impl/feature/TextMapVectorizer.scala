/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Converts a sequence of KeyString features into a vector keeping the top K most common occurrences of each
 * key in the maps for that feature (ie the final vector has length k * number of keys * number of features).
 * Each key found will also generate an other column which will capture values that do not make the cut or where not
 * seen in training. Note that any keys not seen in training will be ingnored.
 *
 * @param uid uid for instance
 */
class TextMapVectorizer[T <: OPMap[String]]
(
  uid: String = UID[TextMapVectorizer[T]]
)(implicit tti: TypeTag[T])
  extends SequenceEstimator[T, OPVector](operationName = "vecTextMap", uid = uid)
    with VectorizerDefaults with PivotParams with MapPivotParams with TextParams
    with MapStringPivotHelper with CleanTextMapFun with MinSupportParam {

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = $(cleanText)

    def convertToMapOfMaps(mapIn: Map[String, String]): MapMap = {
      mapIn.map { case (k, v) => k -> Map(v -> 1L) }
    }

    val categoryMaps: Dataset[SeqMapMap] =
      getCategoryMaps(dataset, convertToMapOfMaps, shouldCleanKeys, shouldCleanValues)

    val topValues: SeqSeqTupArr = getTopValues(categoryMaps, inN.length, $(topK), $(minSupport))

    val vectorMeta = createOutputVectorMetadata(topValues, inN, operationName, outputName)
    setMetadata(vectorMeta.toMetadata)

    new TextMapVectorizerModel[T](
      topValues = topValues,
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues,
      operationName = operationName,
      uid = uid
    )
  }

}

private final class TextMapVectorizerModel[T <: OPMap[String]]
(
  val topValues: Seq[Seq[(String, Array[String])]],
  val shouldCleanKeys: Boolean,
  val shouldCleanValues: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults with CleanTextMapFun {

  def transformFn: Seq[T] => OPVector = row => {
    // Combine top values for each feature with map feature
    val eachPivoted =
      row.zip(topValues).map { case (map, topMap) =>
        val cleanedMap = cleanMap(map.value, shouldCleanKeys, shouldCleanValues)
        topMap.map { case (mapKey, top) =>
          val sizeOfVector = top.length
          cleanedMap.get(mapKey) match {
            case None => Seq(sizeOfVector -> 0.0)
            case Some(cv) =>
              top.indexOf(cv) match {
                case i if i < 0 => Seq(sizeOfVector -> 1.0)
                case i => Seq(i -> 1.0, sizeOfVector -> 0.0)
              }
          }
        }
      }
    // Fix indices for sparse vector
    val reindexed = reindex(eachPivoted.map(reindex))
    makeSparseVector(reindexed).toOPVector
  }
}
