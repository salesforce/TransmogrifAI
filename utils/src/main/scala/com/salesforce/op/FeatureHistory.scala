/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op

import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

/**
 * History of all stages and origin features used to create a given feature
 *
 * @param originFeatures alphabetically ordered names of the raw features this feature was created from
 * @param stages         sequence of the stageNames applied
 */
case class FeatureHistory(originFeatures: Seq[String], stages: Seq[String]) extends JsonLike {

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
