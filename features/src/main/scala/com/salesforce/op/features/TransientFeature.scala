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

package com.salesforce.op.features


import com.salesforce.op.FeatureHistory
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.json4s.JsonAST.JValue
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods
import org.json4s.jackson.JsonMethods.{compact, render}
import org.json4s.{DefaultFormats, Formats}

import scala.util.Try

/**
 * Proxy class to be used by Op Stages that holds a valid reference to a
 * feature on the master node without having to serialize the entire
 * Feature to worker nodes. Only the data that the worker might need is
 * serialized, eg name, isResponse, and isRaw
 *
 * @param name           name of feature
 * @param isResponse     is response feature
 * @param isRaw          is a raw feature
 * @param uid            uid of feature
 * @param typeName       type of feature
 * @param originFeatures raw features used to create this feature
 * @param stages         operation names of all stages used to create this feature
 * @param feature        actual feature used to create transient (only available on driver)
 */

class TransientFeature
(
  val name: String,
  val isResponse: Boolean,
  val isRaw: Boolean,
  val uid: String,
  val typeName: String,
  val originFeatures: Seq[String],
  val stages: Seq[String],
  @transient private var feature: OPFeature = null
) extends Serializable {

  @transient implicit val formats: Formats = DefaultFormats

  def this(f: OPFeature, h: FeatureHistory) = this(
    name = f.name,
    isResponse = f.isResponse,
    isRaw = f.isRaw,
    uid = f.uid,
    typeName = f.typeName,
    originFeatures = h.originFeatures,
    stages = h.stages,
    feature = f
  )

  /**
   * Return the underlying FeatureLike[_] instance
   *
   * @throws RuntimeException in case the feature is null
   * @return FeatureLike[I] instance
   */
  def getFeature(): OPFeature = {
    if (feature == null) {
      throw new RuntimeException(
        s"${this.getClass.getSimpleName}[$name]: feature is null, " +
          "possibly because you are trying to materialize it somewhere other than the driver"
      )
    }
    feature
  }

  /**
   * Return the underlying FeatureLike[_] object cast to feature type I
   *
   * @throws RuntimeException in case the feature is null
   *
   * @tparam I feature type
   * @return FeatureLike[I] instance
   */
  def asFeatureLike[I <: FeatureType]: FeatureLike[I] = getFeature().asInstanceOf[FeatureLike[I]]


  /**
   * Transform trasient feature into column metadata for use vectors
   * (for when each feature creates one column of a vector)
   * @param isNull is the metadata created for a null indicator column
   * @return OpVectorColumnMetadata for vector feature
   */
  def toColumnMetaData(isNull: Boolean = false): OpVectorColumnMetadata = {
    new OpVectorColumnMetadata(
      parentFeatureName = Seq(name),
      parentFeatureType = Seq(typeName),
      indicatorGroup = if (isNull) Some(name) else None,
      indicatorValue = if (isNull) Some(OpVectorColumnMetadata.NullString) else None)
    }

  /**
   * Transform transient feature into vector metadata for use vectors
   * (for when each feature creates multiple columns of a vector) assigns indicator group to feature name since
   * multiple columns are created from single feature - does not provide indicator values since that needs the transform
   * information (can copy this output with known values when available)
   *
   * @param fieldName name of output
   * @param size size of vector being created
   * @return OpVectorMetadata for vector feature
   */
  def toVectorMetaData(size: Int, fieldName: Option[String] = None): OpVectorMetadata = {
    val columns = (0 until size)
      .map{ i => toColumnMetaData().copy(indicatorGroup = Option(name)) }
      .toArray
    val history = Map(name -> FeatureHistory(originFeatures = originFeatures, stages = stages))
    OpVectorMetadata(fieldName.getOrElse(name), columns, history)
  }

  /**
   * Convert to JObject representation without saving the underlying FeatureLike instance
   *
   * @return JObject
   */
  def toJson: JValue = {
    ("name" -> name) ~
      ("isResponse" -> isResponse) ~
      ("isRaw" -> isRaw) ~
      ("uid" -> uid) ~
      ("typeName" -> typeName) ~
      ("originFeatures" -> originFeatures) ~
      ("stages" -> stages)
  }

  /**
   * Convert this instance to json string
   *
   * @return json string
   */
  def toJsonString(pretty: Boolean = true): String = {
    val json = render(toJson)
    if (pretty) JsonMethods.pretty(json) else compact(json)
  }
}

object TransientFeature {
  @transient implicit val formats: Formats = DefaultFormats

  /**
   * Construct instance of TransientFeature from FeatureLike[_]
   *
   * @param f feature to be wrapped
   * @return TransientFeature
   */
  def apply(f: OPFeature): TransientFeature = new TransientFeature(f, f.history())

  /**
   * Construct instance of TransientFeature from raw data. FeatureLike[_] instance is set to null.
   *
   * @param name           name of feature
   * @param isResponse     is response feature
   * @param isRaw          is a raw feature
   * @param uid            uid of feature
   * @param typeName       type of feature
   * @param originFeatures raw features used to create this feature
   * @param stages         operation names of all stages used to create this feature
   * @return TransientFeature
   */
  def apply
  (
    name: String,
    isResponse: Boolean,
    isRaw: Boolean,
    uid: String,
    typeName: String,
    originFeatures: Seq[String],
    stages: Seq[String]
  ): TransientFeature = new TransientFeature(
    name = name,
    isResponse = isResponse,
    isRaw = isRaw,
    uid = uid,
    typeName = typeName,
    originFeatures = originFeatures,
    stages = stages
  )

  /**
   * Construct instance of TransientFeature from JValue. Private use for OP only
   *
   * @param jValue
   * @return
   */
  private[op] def apply(jValue: JValue): Try[TransientFeature] = Try {
    implicit val formats = DefaultFormats
    val typeNameStr = (jValue \ "typeName").extract[String]
    val typeName = FeatureType.typeName(FeatureType.featureTypeTag(typeNameStr))
    new TransientFeature(
      name = (jValue \ "name").extract[String],
      isResponse = (jValue \ "isResponse").extract[Boolean],
      isRaw = (jValue \ "isRaw").extract[Boolean],
      uid = (jValue \ "uid").extract[String],
      typeName = typeName,
      originFeatures = (jValue \ "originFeatures").extract[Seq[String]],
      stages = (jValue \ "stages").extract[Seq[String]]
    )
  }
}
