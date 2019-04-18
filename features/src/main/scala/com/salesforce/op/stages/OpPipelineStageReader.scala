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

package com.salesforce.op.stages

import com.salesforce.op.features.OPFeature
import com.salesforce.op.stages.OpPipelineStageReadWriteShared._
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.SparkDefaultParamsReadWrite
import org.apache.spark.ml.util.MLReader
import org.json4s.JsonAST.JValue
import org.json4s._
import org.json4s.jackson.JsonMethods.{compact, render}

import scala.util.{Failure, Success, Try}

/**
 * Reads the serialized output of [[OpPipelineStageWriter]]
 *
 * @param originalStage original stage instance from the workflow
 * @param features      features loaded so far (used to lookup input features fot the stage)
 */
final class OpPipelineStageReader private
(
  val originalStage: Option[OpPipelineStageBase],
  val features: Seq[OPFeature]
) extends MLReader[OpPipelineStageBase] {

  @deprecated("Stage serialization has to be done without original stage", "0.6.0")
  def this(origStage: OpPipelineStageBase) =
    this(Option(origStage), origStage.getInputFeatures().flatMap(_.allFeatures))

  def this(feats: Seq[OPFeature]) = this(None, feats)

  /**
   * Load from disk. File should contain data serialized in json format
   *
   * @param path to the stored output
   * @return OpPipelineStageBase
   */
  override def load(path: String): OpPipelineStageBase = {
    val metadataPath = new Path(path, "metadata").toString
    loadFromJsonString(sc.textFile(metadataPath, 1).first(), path)
  }

  /**
   * Loads from the json serialized data
   *
   * @param json json
   * @param path to the stored output
   * @return OpPipelineStageBase
   */
  def loadFromJson(json: JValue, path: String): OpPipelineStageBase =
    loadFromJsonString(jsonStr = compact(render(json)), path = path)

  /**
   * Loads from the json serialized data
   *
   * @param jsonStr json string
   * @param path    to the stored output
   * @return OpPipelineStageBase
   */
  def loadFromJsonString(jsonStr: String, path: String): OpPipelineStageBase = {
    // Load stage json with it's params
    val metadata = SparkDefaultParamsReadWrite.parseMetadata(jsonStr)
    val (className, metadataJson) = metadata.className -> metadata.metadata

    // Check if it's a model
    val isModelOpt = (metadataJson \ FieldNames.IsModel.entryName).extractOpt[Boolean]
    val ctorArgsJson = metadataJson \ FieldNames.CtorArgs.entryName
    val stageClass = ReflectionUtils.classForName(className).asInstanceOf[Class[OpPipelineStageBase]]

    val stageTry: Try[OpPipelineStageBase] = isModelOpt match {
      // Legacy mode
      case Some(isModel) =>
        // In case we stumbled upon model we instantiate it using the class name + ctor args
        // otherwise we simply use the provided stage instance.
        if (isModel) new DefaultOpPipelineStageJsonReaderWriter[OpPipelineStageBase]().read(stageClass, ctorArgsJson)
        else originalStage.map(Success(_)).getOrElse(Failure(new RuntimeException("Origin stage was not set")))
      case _ =>
        // Get the reader instance to load the stage
        val reader = readerWriterFor[OpPipelineStageBase](stageClass)
        reader.read(stageClass, ctorArgsJson)
    }
    val stage = stageTry match {
      case Failure(err) => throw new RuntimeException(s"Failed to read the stage of type '${stageClass.getName}'", err)
      case Success(stg) => stg
    }

    // Update [[SparkWrapperParams]] with path so we can load the [[SparkStageParam]] instance
    val updatedMetadata = stage match {
      case _: SparkWrapperParams[_] =>
        val updatedParams = SparkStageParam.updateParamsMetadataWithPath(metadata.params, path)
        metadata.copy(params = updatedParams)
      case _ => metadata
    }

    // Set all stage params from the metadata
    SparkDefaultParamsReadWrite.getAndSetParams(stage, updatedMetadata)

    // Recover and set all stage input features
    val matchingFeatures = stage.getTransientFeatures().map { f =>
      features.find(i => i.uid == f.uid && i.isResponse == f.isResponse && i.typeName == f.typeName)
        .getOrElse(throw new RuntimeException(s"Feature '${f.uid}' was not found for stage '${stage.uid}'"))
    }
    stage.setInputFeatureArray(matchingFeatures)
  }

}
