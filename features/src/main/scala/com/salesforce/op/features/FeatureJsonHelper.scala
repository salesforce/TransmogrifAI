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

package com.salesforce.op.features

import com.salesforce.op.features.types._
import com.salesforce.op.stages.{OPStage, OpPipelineStage}
import com.salesforce.op.utils.json.JsonUtils
import org.apache.spark.sql.types.Metadata
import org.json4s.JsonAST.{JObject, JValue}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, Formats, JArray}

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Try


/**
 * Feature Json helper class allowing to/from Json marshalling of a single feature instance
 */
object FeatureJsonHelper {

  implicit val jsonFormats: Formats = DefaultFormats

  /**
   * Convert this feature to json object
   *
   * @return json object for feature
   */
  def toJson(f: OPFeature): JObject = {
    ("typeName" -> f.typeName) ~
      ("uid" -> f.uid) ~
      ("name" -> f.name) ~
      ("isResponse" -> f.isResponse) ~
      ("originStage" -> f.originStage.uid) ~
      ("parents" -> f.parents.map(_.uid)) ~
      ("distributions" -> f.distributions.map(JsonUtils.toJsonString(_, false))) ~
      ("metadata" -> f.metadata.map(JsonUtils.toJsonString(_, false)))
  }

  /**
   * Convert this feature to json string
   *
   * @param pretty should pretty print
   * @return json string for feature
   */
  def toJsonString(f: OPFeature, pretty: Boolean = false): String = {
    val json = render(toJson(f))
    if (pretty) JsonMethods.pretty(json) else compact(json)
  }

  /**
   * Convert json back into a feature instance
   *
   * @param json     a json string of a single feature
   * @param stages   a map of all op pipeline stages by uid
   * @param features a map of all op pipeline features by uid
   *
   * @return new feature instance
   */
  def fromJsonString(
    json: String,
    stages: Map[String, OPStage],
    features: Map[String, OPFeature]
  ): Try[OPFeature] = {
    for {
      json <- Try(parse(json).asInstanceOf[JObject])
      feature <- fromJson(json, stages, features)
    } yield feature
  }

  /**
   * Convert json back into a feature instance
   *
   * @param json     a json value of a single feature
   * @param stages   a map of all op pipeline stages by uid
   * @param features a map of all op pipeline features by uid
   *
   * @return new feature instance
   */
  def fromJson(
    json: JValue,
    stages: Map[String, OPStage],
    features: Map[String, OPFeature]
  ): Try[OPFeature] = Try {
    val typeName = (json \ "typeName").extract[String]
    val uid = (json \ "uid").extract[String]
    val name = (json \ "name").extract[String]
    val isResponse = (json \ "isResponse").extract[Boolean]
    val originStageUid = (json \ "originStage").extract[String]
    val parentUids = (json \ "parents").extract[Array[String]]
    // TODO get this to work
    // val distributions = (json \ "distributions").extract[Array[String]]
    //  .flatMap(JsonUtils.fromString[FeatureDistributionLike](_).toOption)
    val metadata = (json \ "metadata").extractOpt[String]
      .flatMap(JsonUtils.fromString[Metadata](_).toOption)

    val originStage: Option[OPStage] = stages.get(originStageUid)
    if (originStage.isEmpty) {
      throw new RuntimeException(s"Origin stage $originStageUid not found for feature $name ($uid)")
    }

    // Order is important and so are duplicates, eg f = f1 + f1 has 2 parents but both the same feature
    val parents: Seq[OPFeature] = parentUids.flatMap(id => features.get(id))
    if (parents.length != parentUids.length) {
      throw new RuntimeException(s"Not all the parent features were found for feature $name ($uid)")
    }

    val wtt = FeatureType.featureTypeTag(typeName).asInstanceOf[WeakTypeTag[FeatureType]]
    Feature[FeatureType](
      uid = uid,
      name = name,
      isResponse = isResponse,
      parents = parents,
      originStage = originStage.get.asInstanceOf[OpPipelineStage[FeatureType]],
      distributions = Seq.empty,
      metadata = metadata
    )(wtt = wtt)

  }

}
