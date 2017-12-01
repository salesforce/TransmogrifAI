/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe._

/**
 * Converts a sequence of KeyMultiPickList features into a vector keeping the top K most common occurrences of each
 * key in the maps for that feature (ie the final vector has length k * number of keys * number of features).
 * Each key found will also generate an other column which will capture values that do not make the cut or where not
 * seen in training. Note that any keys not seen in training will be ignored.
 *
 * @param uid uid for instance
 */
class MultiPickListMapVectorizer[T <: OPMap[Set[String]]]
(
  uid: String = UID[MultiPickListMapVectorizer[T]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends SequenceEstimator[T, OPVector](operationName = "vecCatMap", uid = uid)
  with VectorizerDefaults with PivotParams with MapPivotParams with TextParams
  with MapStringPivotHelper with CleanTextMapFun with MinSupportParam {

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = $(cleanText)

    def convertToMapOfMaps(mapIn: Map[String, Set[String]]): MapMap = {
      mapIn.map { case (k, cats) =>
        k -> cats.map(_ -> 1L).groupBy(_._1).map { case (c, a) => c -> a.map(_._2).sum }
      }
    }

    val categoryMaps: Dataset[SeqMapMap] =
      getCategoryMaps(dataset, convertToMapOfMaps, shouldCleanKeys, shouldCleanValues)

    val topValues: Seq[Seq[(String, Array[String])]] = getTopValues(categoryMaps, inN.length, $(topK), $(minSupport))

    val vectorMeta = createOutputVectorMetadata(topValues, inN, operationName, outputName)
    setMetadata(vectorMeta.toMetadata)

    new MultiPickListMapVectorizerModel(
      topValues = topValues, shouldCleanKeys = shouldCleanKeys, shouldCleanValues = shouldCleanValues,
      operationName = operationName, uid = uid
    )
  }

}

private final class MultiPickListMapVectorizerModel[T <: OPMap[Set[String]]]
(
  val topValues: Seq[Seq[(String, Array[String])]],
  val shouldCleanKeys: Boolean,
  val shouldCleanValues: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults with CleanTextMapFun {

  def transformFn: (Seq[T]) => OPVector = row => {
    // Combine top values for each feature with map feature
    val eachPivoted =
      row.zip(topValues).map { case (map, topMap) =>
        val cleanedMap = cleanMap(map.value, shouldCleanKeys, shouldCleanValues)
        topMap.map { case (mapKey, top) =>
          val sizeOfVector = top.length
          cleanedMap.get(mapKey) match {
            case None => Seq(sizeOfVector -> 0.0)
            case Some(mapVal) =>
              val topPivotVals =
                mapVal
                  .map(top.indexOf)
                  .collect { case i if i >= 0 => i -> 1.0 }
                  .toSeq
              if (topPivotVals.isEmpty) Seq(sizeOfVector -> 1.0)
              else topPivotVals ++ Seq(sizeOfVector -> 0.0)
          }
        }
      }
    // Fix indices for sparse vector
    val reindexed = reindex(eachPivoted.map(reindex))
    makeSparseVector(reindexed).toOPVector
  }

}
