/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.hadoop.fs.Path
import OpPipelineStageReadWriteShared._
import org.apache.spark.ml.SparkDefaultParamsReadWrite
import org.apache.spark.ml.util.MLReader
import org.json4s.Extraction
import org.json4s.JsonAST.{JObject, JValue}
import org.json4s.jackson.JsonMethods.{compact, render}

import scala.reflect.runtime.universe._
import scala.util.{Failure, Success, Try}

/**
 * Reads the serialized output of [[OpPipelineStageWriter]]
 *
 * @param originalStage original serialized stage
 */
final class OpPipelineStageReader(val originalStage: OpPipelineStageBase)
  extends MLReader[OpPipelineStageBase] {

  /**
   * Load from disk. File should contain data serialized in json format
   *
   * @param path to the stored output
   * @return OpPipelineStageBase
   */
  override def load(path: String): OpPipelineStageBase = {
    val metadataPath = new Path(path, "metadata").toString
    loadFromJsonString(sc.textFile(metadataPath, 1).first())
  }

  /**
   * Loads from the json serialized data
   *
   * @param json json
   * @return OpPipelineStageBase
   */
  def loadFromJson(json: JValue): OpPipelineStageBase = loadFromJsonString(compact(render(json)))

  /**
   * Loads from the json serialized data
   *
   * @param jsonStr json string
   * @return OpPipelineStageBase
   */
  def loadFromJsonString(jsonStr: String): OpPipelineStageBase = {
    // Load stage json with it's params
    val metadata = SparkDefaultParamsReadWrite.parseMetadata(jsonStr)
    val (className, metadataJson) = metadata.className -> metadata.metadata
    // Check if it's a model
    val isModel = (metadataJson \ FieldNames.IsModel.entryName).extract[Boolean]
    // In case we stumbled upon model we instantiate it using the class name + ctor args
    // otherwise we simply use the provided stage instance.
    val stage = if (isModel) loadModel(className, metadataJson) else originalStage

    // Recover all stage spark params and it's input features
    val inputFeatures = originalStage.getInputFeatures()
    SparkDefaultParamsReadWrite.getAndSetParams(stage, metadata)
    val matchingFeatures = stage.getTransientFeatures().map{ f =>
      inputFeatures.find( i => i.uid == f.uid && i.isResponse == f.isResponse && i.typeName == f.typeName )
        .getOrElse( throw new RuntimeException(s"Feature '${f.uid}' was not found for stage '${stage.uid}'") )
    }
    stage.setInputFeatureArray(matchingFeatures)
  }

  /**
   * Load the model instance from the metadata by instantiating it using a class name + ctor args
   */
  private def loadModel(modelClassName: String, metadataJson: JValue): OpPipelineStageBase = {
    // Extract all the ctor args
    val ctorArgsJson = (metadataJson \ FieldNames.CtorArgs.entryName).asInstanceOf[JObject].obj
    val ctorArgsMap = ctorArgsJson.map { case (argName, j) => argName -> j.extract[AnyValue] }.toMap
    // Get the model class

    // Make the ctor function used for creating a model instance
    def ctorArgs(argName: String, argSymbol: Symbol): Try[Any] = Try {
      val anyValue = ctorArgsMap.getOrElse(argName,
        throw new RuntimeException(s"Ctor argument '$argName' was not found for model class '$modelClassName'")
      )
      anyValue.`type` match {
        // Special handling for Feature Type TypeTags
        case AnyValueTypes.TypeTag =>
          Try(FeatureType.featureTypeTag(anyValue.value.toString)).recoverWith[TypeTag[_]] { case e =>
            Try(FeatureType.featureValueTypeTag(anyValue.value.toString))
          } match {
            case Success(featureTypeTag) => featureTypeTag
            case Failure(e) =>
              throw new RuntimeException(
                s"Unknown type tag '${anyValue.value.toString}' for ctor argument '$argName'. " +
                  "Only Feature and Feature Value type tags are supported for serialization.", e
              )
          }

        // Spark wrapped stage is saved using [[SparkWrapperParams]] and loaded later using
        // [[DefaultParamsReader]].getAndSetParams returning 'null' here
        case AnyValueTypes.SparkWrappedStage => {
          null // yes, yes - this should be 'null'
        }

        // Everything else is read using json4s
        case AnyValueTypes.Value => Try {
          val ttag = ReflectionUtils.typeTagForType[Any](tpe = argSymbol.info)
          val manifest = ReflectionUtils.manifestForTypeTag[Any](ttag)
          Extraction.decompose(anyValue.value).extract[Any](formats, manifest)
        } match {
          case Success(any) => any
          case Failure(e) => throw new RuntimeException(
            s"Failed to parse argument '$argName' from value '${anyValue.value}'", e)
        }
      }
    }

    // Reflect model class instance by class + ctor args
    val modelClass = ReflectionUtils.classForName(modelClassName)
    val model = ReflectionUtils.newInstance[OpPipelineStageBase](modelClass, ctorArgs)
    model
  }
}
