/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata, SequenceAggregators}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.DoubleArrayParam
import org.apache.spark.sql.{Dataset, Encoders}


class GeolocationMapVectorizer
(
  operationName: String = "vecGeoMap",
  uid: String = UID[GeolocationMapVectorizer]
) extends SequenceEstimator[GeolocationMap, OPVector](operationName = operationName, uid = uid)
  with VectorizerDefaults with MapPivotParams with CleanTextMapFun {
  private implicit val seqSetEncoder = Encoders.kryo[Seq[Set[String]]]
  private implicit val seqArrayEncoder = Encoders.kryo[Seq[Array[Double]]]

  final val defaultValue = new DoubleArrayParam(
    parent = this, name = "defaultValue", doc = "value to give missing keys when pivoting"
  )
  setDefault(defaultValue, Transmogrifier.DefaultGeolocation.toArray)
  def setDefaultValue(value: Geolocation): this.type = set(defaultValue, value.toArray)

  private def getKeyValues(in: Dataset[Seq[GeolocationMap#Value]], shouldClean: Boolean): Seq[Seq[String]] = {
    val inputSize = getInputFeatures().length
    in.map(_.map(kb => filterKeys(kb, shouldClean, shouldCleanValue = false).keySet))
      .select(SequenceAggregators.SumSeqSet(size = inputSize))
      .first()
      .map(_.toSeq)
  }

  private def makeVectorMeta(allKeys: Seq[Seq[String]]): OpVectorMetadata = {
    val meta = vectorMetadataFromInputFeatures
    val cols = for {
      (keys, col) <- allKeys.zip(meta.columns)
      key <- keys
    } yield new OpVectorColumnMetadata(
      parentFeatureName = col.parentFeatureName,
      parentFeatureType = col.parentFeatureType,
      indicatorGroup = None,
      indicatorValue = Option(key)
    )
    meta.withColumns(cols.toArray)
  }

  def fitFn(dataset: Dataset[Seq[GeolocationMap#Value]]): SequenceModel[GeolocationMap, OPVector] = {
    val shouldClean = $(cleanKeys)
    val defValue = $(defaultValue).toSeq
    val allKeys = getKeyValues(dataset, shouldClean)
    setMetadata(makeVectorMeta(allKeys).toMetadata)

    new GeolocationMapVectorizerModel(
      allKeys = allKeys, defaultValue = defValue, shouldClean = shouldClean, operationName = operationName, uid = uid
    )
  }

}

private final class GeolocationMapVectorizerModel
(
  val allKeys: Seq[Seq[String]],
  val defaultValue: Seq[Double],
  val shouldClean: Boolean,
  operationName: String,
  uid: String
) extends SequenceModel[GeolocationMap, OPVector](operationName = operationName, uid = uid)
  with CleanTextMapFun {

  def transformFn: Seq[GeolocationMap] => OPVector = row => {
    val eachPivoted: Array[Array[Double]] =
      row.map(_.value).zip(allKeys).flatMap { case (map, keys) =>
        val cleanedMap = cleanMap(map, shouldClean, shouldCleanValue = false)
        keys.map(k => cleanedMap.getOrElse(k, defaultValue).toArray)
      }.toArray
    Vectors.dense(eachPivoted.flatten).compressed.toOPVector
  }
}
