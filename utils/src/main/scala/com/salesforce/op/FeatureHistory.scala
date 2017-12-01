/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

/**
 * History of all stages and origin features used to create a given feature
 *
 * @param originFeatures alphabetically ordered names of the raw features this feature was created from
 * @param stages         sequence of the operation names applied
 */

case class FeatureHistory
(
  originFeatures: Seq[String],
  stages: Seq[String]
) extends JsonLike {

  /**
   * Convert to Spark metadata
   *
   * @return metadata representation
   */
  def toMetadata: Metadata = {
    new MetadataBuilder()
      .putStringArray(FeatureHistory.OriginFeatureKey, originFeatures.toArray)
      .putStringArray(FeatureHistory.StagesKey, stages.toArray)
      .build()
  }

  /**
   * Combine feature histories from multiple features - Note that stage order will not reflect order applied after this
   * @param other other feature histories
   * @return combine history of all features
   */
  def merge(other: FeatureHistory*): FeatureHistory = {
    val originCombined = (originFeatures ++ other.flatMap(_.originFeatures)).distinct.sorted
    val stagesCombined = (stages ++ other.flatMap(_.stages)).distinct.sorted
    FeatureHistory(originCombined, stagesCombined)
  }
}


case object FeatureHistory {

  val OriginFeatureKey = "origin_features"
  val StagesKey = "stages"

  /**
   * Build metadata from Map of [[FeatureHistory]] instances
   *
   * @param map Map from feature name to [[FeatureHistory]] of that feature
   * @return metadata representation
   */
  def toMetadata(map: Map[String, FeatureHistory]): Metadata = {
    val builder = new MetadataBuilder()
    map.foreach { case (k, v) => builder.putMetadata(k, v.toMetadata) }
    builder.build()
  }

  /**
   * Build Map of [[FeatureHistory]] instances from metadata
   *
   * @param meta metadata containing a mapping from feature name to [[FeatureHistory]]
   * @return map of that inforamtion
   */
  def fromMetadataMap(meta: Metadata): Map[String, FeatureHistory] = {
    val historyMap = meta.wrapped.underlyingMap
    historyMap.map { case (k, v) => k -> fromMetadata(v.asInstanceOf[Metadata]) }
  }

  /**
   * Build [[FeatureHistory]] from metadate
   *
   * @param meta Metadata representing [[FeatureHistory]] contain [[OriginFeatureKey]] and [[StagesKey]]
   * @return new insttance of [[FeatureHistory]]
   */
  def fromMetadata(meta: Metadata): FeatureHistory = {
    val wrapped = meta.wrapped
    FeatureHistory(
      originFeatures = wrapped.getArray[String](OriginFeatureKey),
      stages = wrapped.getArray[String](StagesKey)
    )
  }

}
