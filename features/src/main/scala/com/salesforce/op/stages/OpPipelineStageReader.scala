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
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.OpPipelineStageReadWriteShared._
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.SparkDefaultParamsReadWrite
import org.apache.spark.ml.util.MLReader
import org.json4s.JsonAST.{JObject, JValue}
import org.json4s.jackson.JsonMethods.{compact, render}
import org.json4s.{Extraction, _}

import scala.reflect.ManifestFactory
import scala.reflect.runtime.universe._
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
   * @param path to the stored output
   * @return OpPipelineStageBase
   */
  def loadFromJsonString(jsonStr: String, path: String): OpPipelineStageBase = {
    // Load stage json with it's params
    val metadata = SparkDefaultParamsReadWrite.parseMetadata(jsonStr)
    val (className, metadataJson) = metadata.className -> metadata.metadata
    // Check if it's a model
    val isModelOpt = (metadataJson \ FieldNames.IsModel.entryName).extractOpt[Boolean]

    val stage = isModelOpt match {
      // Legacy mode
      case Some(isModel) =>
        // In case we stumbled upon model we instantiate it using the class name + ctor args
        // otherwise we simply use the provided stage instance.
        if (isModel) loadStage(className, metadataJson)
        else originalStage.getOrElse(throw new RuntimeException("Origin stage was not set"))
      case _ =>
        loadStage(className, metadataJson)
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

  /**
   * Load the stage instance from the metadata by instantiating it using a class name + ctor args
   */
  private def loadStage(stageClassName: String, metadataJson: JValue): OpPipelineStageBase = {
    // Extract all the ctor args
    val ctorArgsJson = (metadataJson \ FieldNames.CtorArgs.entryName).asInstanceOf[JObject].obj
    val ctorArgsMap = ctorArgsJson.map { case (argName, j) => argName -> j.extract[AnyValue] }.toMap

    // Make the ctor function used for creating a stage instance
    def ctorArgs(argName: String, argSymbol: Symbol): Try[Any] = {
      for {
        anyValue <- Try {
          ctorArgsMap.getOrElse(argName,
            throw new RuntimeException(s"Ctor argument '$argName' was not found for stage class '$stageClassName'"))
        }
        argInstance = Try {
          anyValue match {
            // Special handling for Feature Type TypeTags
            case AnyValue(AnyValueTypes.TypeTag, value, _) =>
              Try(FeatureType.featureTypeTag(value.toString)).recoverWith[TypeTag[_]] { case _ =>
                Try(FeatureType.featureValueTypeTag(value.toString))
              } match {
                case Success(featureTypeTag) => featureTypeTag
                case Failure(e) => throw new RuntimeException(
                  s"Unknown type tag '${value.toString}' for ctor argument '$argName'. " +
                    "Only Feature and Feature Value type tags are supported for serialization.", e)
              }

            // Spark wrapped stage is saved using [[SparkWrapperParams]] and loaded later using
            // [[SparkDefaultParamsReadWrite]].getAndSetParams returning 'null' here
            case AnyValue(AnyValueTypes.SparkWrappedStage, _, _) => null // yes, yes - this should be 'null'

            // Class value argument, e.g. [[Function1]], [[Numeric]] etc.
            case AnyValue(AnyValueTypes.ClassInstance, value, _) =>
              ReflectionUtils.classForName(value.toString).getConstructors.head.newInstance()

            // Simple value with no ctor argument should be instantiable by class name
            case AnyValue(AnyValueTypes.Value, m: Map[_, _], Some(className)) if m.isEmpty =>
              ReflectionUtils.classForName(className).getConstructors.head.newInstance()

            // Everything else is read using json4s
            case AnyValue(AnyValueTypes.Value, value, valueClass) =>
              // Create type manifest either using the reflected type tag or serialized value class
              val manifest = try {
                val ttag = ReflectionUtils.typeTagForType[Any](tpe = argSymbol.info)
                ReflectionUtils.manifestForTypeTag[Any](ttag)
              } catch {
                case _ if valueClass.isDefined =>
                  ManifestFactory.classType[Any](ReflectionUtils.classForName(valueClass.get))
              }
              Extraction.decompose(value).extract[Any](formats, manifest)

          }
        }
        res <- argInstance match {
          case Failure(e) =>
            throw new RuntimeException(
              s"Failed to parse argument '$argName' from value '${anyValue.value}'" +
                anyValue.valueClass.map(c => s" of class '$c'").getOrElse(""), e)
          case ok => ok
        }
      } yield res
    }

    // Reflect stage class instance by class + ctor args
    val stageClass = ReflectionUtils.classForName(stageClassName)
    val stage = ReflectionUtils.newInstance[OpPipelineStageBase](stageClass, ctorArgs)
    stage
  }

}
