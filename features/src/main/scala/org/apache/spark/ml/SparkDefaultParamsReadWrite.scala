// scalastyle:off header.matches
/*
 * Modifications: (c) 2017, Salesforce.com, Inc.
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml

import com.salesforce.op.stages.OpPipelineStageBase
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter}
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._

/**
 * Direct wrappers for ml private [[DefaultParamsWriter]] and [[DefaultParamsReader]]
 * needed to read/write Spark stages in OP
 */
case object SparkDefaultParamsReadWrite {

  type Metadata = DefaultParamsReader.Metadata

  /**
   * Helper for [[OpPipelineStageWriter]] which extracts the JSON to save.
   * This is useful for ensemble models which need to save metadata for many sub-models.
   *
   * Note: this method was taken from DefaultParamsWriter.getMetadataToSave,
   * but modified to avoid requiring Spark session instead use `org.apache.spark.SPARK_VERSION`
   *
   * @see [[OpPipelineStageWriter]] for details on what this includes.
   */
  def getMetadataToSave(
    instance: OpPipelineStageBase,
    extraMetadata: Option[JObject] = None,
    paramMap: Option[JValue] = None
  ): String = {
    val uid = instance.uid
    val cls = instance.getClass.getName
    val params = instance.paramMap.toSeq
    val defaultParams = instance.defaultParamMap.toSeq
    val jsonParams = paramMap.getOrElse(render(params.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))
    val jsonDefaultParams = render(defaultParams.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList)
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> org.apache.spark.SPARK_VERSION) ~
      ("uid" -> uid) ~
      ("paramMap" -> jsonParams) ~
      ("defaultParamMap" -> jsonDefaultParams)
    val metadata = extraMetadata match {
      case Some(jObject) =>
        basicMetadata ~ jObject
      case None =>
        basicMetadata
    }
    val metadataJson: String = compact(render(metadata))
    metadataJson
  }

  /**
   * Parse metadata JSON string produced by [[DefaultParamsWriter.getMetadataToSave()]].
   * This is a helper function for [[loadMetadata()]].
   *
   * Note: this method was taken from [[DefaultParamsWriter.parseMetadata]],
   * but modified to avoid failing on loading of Spark models prior to 2.4.x
   *
   * @param metadataStr  JSON string of metadata
   * @param expectedClassName  If non empty, this is checked against the loaded metadata.
   * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
   */
  def parseMetadata(metadataStr: String, expectedClassName: String = ""): Metadata = {
    val metadata = parse(metadataStr)

    implicit val format = DefaultFormats
    val className = (metadata \ "class").extract[String]
    val uid = (metadata \ "uid").extract[String]
    val timestamp = (metadata \ "timestamp").extract[Long]
    val sparkVersion = (metadata \ "sparkVersion").extract[String]
    val params = metadata \ "paramMap"
    val defaultParams = metadata \ "defaultParamMap"
    if (expectedClassName.nonEmpty) {
      require(className == expectedClassName, s"Error loading metadata: Expected class name" +
        s" $expectedClassName but found class name $className")
    }
    // ******************************************************************************************
    /**
     * Backward compatible fix for models trained with older versions of Spark (prior to 2.4.x).
     * The change introduced in https://github.com/apache/spark/pull/20633 added serialization of
     * default params, older models won't have them and fail to load.
     */
    val defaultParamsFix = if (defaultParams == JNothing) JObject() else defaultParams
    // ******************************************************************************************

    new Metadata(className, uid, timestamp, sparkVersion, params, defaultParamsFix, metadata, metadataStr)
  }

  /**
   * Extract Params from metadata, and set them in the instance.
   * This works if all Params (except params included by `skipParams` list) implement
   * [[org.apache.spark.ml.param.Param.jsonDecode()]].
   *
   * @param skipParams The params included in `skipParams` won't be set. This is useful if some
   *                   params don't implement [[org.apache.spark.ml.param.Param.jsonDecode()]]
   *                   and need special handling.
   */
  def getAndSetParams(stage: OpPipelineStageBase, metadata: Metadata, skipParams: Option[List[String]] = None): Unit =
    metadata.getAndSetParams(stage, skipParams)

}
