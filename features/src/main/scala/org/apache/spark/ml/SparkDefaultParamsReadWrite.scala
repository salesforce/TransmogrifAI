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

  implicit val formats = DefaultFormats

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
    stage: OpPipelineStageBase,
    extraMetadata: Option[JObject] = None,
    paramMap: Option[JValue] = None
  ): JObject = {
    val uid = stage.uid
    val cls = stage.getClass.getName
    val params = stage.extractParamMap().toSeq.asInstanceOf[Seq[ParamPair[Any]]]
    val jsonParams = paramMap.getOrElse(render(params.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> org.apache.spark.SPARK_VERSION) ~
      ("uid" -> uid) ~
      ("paramMap" -> jsonParams)
    val metadata = extraMetadata match {
      case Some(jObject) =>
        basicMetadata ~ jObject
      case None =>
        basicMetadata
    }
    metadata
  }

  /**
   * Parse metadata JSON string produced by [[DefaultParamsWriter.getMetadataToSave()]].
   * This is a helper function for [[loadMetadata()]].
   *
   * @param metadataStr  JSON string of metadata
   * @param expectedClassName  If non empty, this is checked against the loaded metadata.
   * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
   */
  def parseMetadata(jsonStr: String): Metadata =
    DefaultParamsReader.parseMetadata(jsonStr)

  /**
   * Extract Params from metadata, and set them in the instance.
   * This works if all Params implement [[org.apache.spark.ml.param.Param.jsonDecode()]].
   * TODO: Move to [[Metadata]] method
   */
  def getAndSetParams(stage: OpPipelineStageBase, metadata: Metadata): Unit =
    DefaultParamsReader.getAndSetParams(stage, metadata)

}
