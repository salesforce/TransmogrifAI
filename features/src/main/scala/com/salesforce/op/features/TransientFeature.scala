/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features


import com.salesforce.op.FeatureHistory
import com.salesforce.op.features.types.FeatureType
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
   * Set the underlying FeatureLike[_] instance.
   * NOTE: feature instance must match the transient feature uid, name typeName etc.
   *
   * @param f FeatureLike[_] instance
   * @throws IllegalArgumentException if feature instance does not match transient feature uid / name / typeName etc.
   * @return this instance
   */
  def setFeature(f: OPFeature): this.type = {
    lazy val history = f.history()
    val requirements = Seq(
      (() => f != null, () => "feature is null"),
      (() => name == f.name, () => s"names do not match: $name != ${f.name}"),
      (() => isResponse == f.isResponse, () => s"feature isResponse value is invalid: $isResponse != ${f.isResponse}"),
      (() => isRaw == f.isRaw, () => s"feature isRaw value is invalid: $isRaw != ${f.isRaw}"),
      (() => uid == f.uid, () => s"UIDs do not match: $uid != ${f.uid}"),
      (() => typeName == f.typeName, () => s"types do not match: $typeName != ${f.typeName}"),
      (() => originFeatures sameElements history.originFeatures,
        () => s"origin features do not match: $originFeatures != ${history.originFeatures}"),
      (() => stages sameElements history.stages,
        () => s"stages do not match: $stages != ${history.stages}")
    )
    requirements.dropWhile(_._1.apply()).headOption.foreach { case (_, error) =>
      throw new IllegalArgumentException(s"Setting feature for transient feature '$uid' failed: " + error.apply())
    }
    this.feature = f
    this
  }

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
