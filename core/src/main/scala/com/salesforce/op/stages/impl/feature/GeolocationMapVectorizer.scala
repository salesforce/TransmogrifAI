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
  with VectorizerDefaults with MapPivotParams with CleanTextMapFun
  with MapVectorizerFuns[Seq[Double], GeolocationMap] with TrackNullsParam {
  private implicit val seqArrayEncoder = Encoders.kryo[Seq[Array[Double]]]

  final val defaultValue = new DoubleArrayParam(
    parent = this, name = "defaultValue", doc = "value to give missing keys when pivoting"
  )
  setDefault(defaultValue, TransmogrifierDefaults.DefaultGeolocation.toArray)

  def setDefaultValue(value: Geolocation): this.type = set(defaultValue, value.toArray)

  override def makeVectorMetadata(allKeys: Seq[Seq[String]]): OpVectorMetadata = {
    val meta = vectorMetadataFromInputFeatures

    val cols = for {
      (keys, col) <- allKeys.zip(meta.columns)
      key <- keys
      // We don't store this in the metadata directly, but need to make 3 cols per key - lat, lon, acc
      index <- Array.range(0, 3)
    } yield new OpVectorColumnMetadata(
      parentFeatureName = col.parentFeatureName,
      parentFeatureType = col.parentFeatureType,
      indicatorGroup = Option(key),
      indicatorValue = None
    )
    meta.withColumns(cols.toArray)
  }

  override def makeVectorMetaWithNullIndicators(allKeys: Seq[Seq[String]]): OpVectorMetadata = {
    val vectorMeta = makeVectorMetadata(allKeys)
    val updatedCols = vectorMeta.columns.grouped(3).flatMap { col => {
      val head = col.head
      col :+ OpVectorColumnMetadata(
        parentFeatureName = head.parentFeatureName,
        parentFeatureType = head.parentFeatureType,
        indicatorGroup = head.indicatorGroup,
        indicatorValue = Some(TransmogrifierDefaults.NullString)
      )
    }
    }.toArray
    vectorMeta.withColumns(updatedCols)
  }

  def fitFn(dataset: Dataset[Seq[GeolocationMap#Value]]): SequenceModel[GeolocationMap, OPVector] = {
    val shouldClean = $(cleanKeys)
    val defValue = $(defaultValue).toSeq
    val allKeys = getKeyValues(dataset, shouldClean, false)
    val trackNullsValue = $(trackNulls)

    val meta = if (trackNullsValue) makeVectorMetaWithNullIndicators(allKeys) else makeVectorMetadata(allKeys)
    setMetadata(meta.toMetadata)

    new GeolocationMapVectorizerModel(
      allKeys = allKeys, defaultValue = defValue, shouldClean = shouldClean, trackNulls = trackNullsValue,
      operationName = operationName, uid = uid
    )
  }

}

final class GeolocationMapVectorizerModel private[op]
(
  val allKeys: Seq[Seq[String]],
  val defaultValue: Seq[Double],
  val shouldClean: Boolean,
  val trackNulls: Boolean,
  operationName: String,
  uid: String
) extends SequenceModel[GeolocationMap, OPVector](operationName = operationName, uid = uid)
  with CleanTextMapFun {

  def transformFn: Seq[GeolocationMap] => OPVector = row => {
    val eachPivoted: Array[Array[Double]] =
      row.map(_.value).zip(allKeys).flatMap { case (map, keys) =>
        val cleanedMap = cleanMap(map, shouldClean, shouldCleanValue = false)
        keys.map(k => {
          val vOpt = cleanedMap.get(k)
          val isEmpty = vOpt.isEmpty
          val v = vOpt.getOrElse(defaultValue).toArray
          if (trackNulls) v :+ (if (isEmpty) 1.0 else 0.0) else v
        })
      }.toArray
    Vectors.dense(eachPivoted.flatten).compressed.toOPVector
  }
}
