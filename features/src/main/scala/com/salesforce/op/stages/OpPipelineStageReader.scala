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

import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.OpPipelineStageReadWriteShared.{FieldNames, _}
import com.salesforce.op.stages.base.LambdaTransformer
import com.salesforce.op.stages.base.binary.BinaryLambdaTransformer
import com.salesforce.op.stages.base.quaternary.QuaternaryLambdaTransformer
import com.salesforce.op.stages.base.sequence.{BinarySequenceLambdaTransformer, SequenceLambdaTransformer}
import com.salesforce.op.stages.base.ternary.TernaryLambdaTransformer
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.SparkDefaultParamsReadWrite
import org.apache.spark.ml.util.MLReader
import org.json4s.JsonAST.{JObject, JValue}
import org.json4s.jackson.JsonMethods.{compact, render}
import org.json4s.{Extraction, _}

import scala.reflect.runtime.universe._
import scala.util.parsing.json.JSONArray
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

  private[this] val ClassUnaryLambdaTransformerName = classOf[UnaryLambdaTransformer[_, _]].getName
  private[this] val ClassSequenceLambdaTransformerName = classOf[SequenceLambdaTransformer[_, _]].getName
  private[this] val ClassBinaryLambdaTransformerName = classOf[BinaryLambdaTransformer[_, _, _]].getName
  private[this] val ClassBinarySequenceLambdaTransformerName = classOf[BinarySequenceLambdaTransformer[_, _, _]].getName
  private[this] val ClassTernaryLambdaTransformerName = classOf[TernaryLambdaTransformer[_, _, _, _]].getName
  private[this] val ClassQuaternaryLambdaTransformerName = classOf[QuaternaryLambdaTransformer[_, _, _, _, _]].getName


  @inline private def getFeatureTypeTag(metadataJson: JValue, fieldName: FieldNames) =
    ReflectionUtils.typeTagForName(n = (metadataJson \ fieldName.entryName).extract[String])
      .asInstanceOf[TypeTag[FeatureType]]


  private def loadLambdaTransformer(metadataJson: JValue, className: String): Option[Try[OpPipelineStageBase]] = {
    val lambdaClassNameOpt = (metadataJson \ FieldNames.LambdaClassName.entryName).extractOpt[String]
    val ctorArgs = loadCtorArgs(metadataJson \ FieldNames.LambdaClassArgs.entryName)
    lambdaClassNameOpt map {
      name =>
        ReflectionUtils.newLambdaInstance(name, ctorArgs) map {
          lambdaInst =>
            val uid = (metadataJson \ FieldNames.Uid.entryName).extract[String]
            val tto = getFeatureTypeTag(metadataJson, FieldNames.LambdaTypeO)
            val tti1 = getFeatureTypeTag(metadataJson, FieldNames.LambdaTypeI1)
            val ttov = FeatureType.featureValueTypeTag((metadataJson \ FieldNames.LambdaTypeOV.entryName)
              .extract[String]).asInstanceOf[TypeTag[FeatureType#Value]]
            val opName = (metadataJson \ FieldNames.OperationName.entryName).extract[String]

            className match {
              case ClassUnaryLambdaTransformerName => {
                new UnaryLambdaTransformer[FeatureType, FeatureType](uid = uid,
                  operationName = opName,
                  transformFn = lambdaInst.asInstanceOf[FeatureType => FeatureType]
                )(tti1, tto, ttov)
              }

              case ClassSequenceLambdaTransformerName => {
                new SequenceLambdaTransformer[FeatureType, FeatureType](uid = uid,
                  operationName = opName,
                  transformFn = lambdaInst.asInstanceOf[Seq[FeatureType] => FeatureType]
                )(tti1, tto, ttov)
              }

              case ClassBinaryLambdaTransformerName => {
                val tti2 = getFeatureTypeTag(metadataJson, FieldNames.LambdaTypeI2)
                new BinaryLambdaTransformer[FeatureType, FeatureType, FeatureType](uid = uid,
                  operationName = opName,
                  transformFn = lambdaInst.asInstanceOf[(FeatureType, FeatureType) => FeatureType]
                )(tti1, tti2, tto, ttov)
              }

              case ClassBinarySequenceLambdaTransformerName => {
                val tti2 = getFeatureTypeTag(metadataJson, FieldNames.LambdaTypeI2)
                new BinarySequenceLambdaTransformer[FeatureType, FeatureType, FeatureType](uid = uid,
                  operationName = opName,
                  transformFn = lambdaInst.asInstanceOf[(FeatureType, Seq[FeatureType]) => FeatureType]
                )(tti1, tti2, tto, ttov)
              }

              case ClassTernaryLambdaTransformerName => {
                val tti2 = getFeatureTypeTag(metadataJson, FieldNames.LambdaTypeI2)
                val tti3 = getFeatureTypeTag(metadataJson, FieldNames.LambdaTypeI3)
                new TernaryLambdaTransformer[FeatureType, FeatureType, FeatureType, FeatureType](uid = uid,
                  operationName = opName,
                  transformFn = lambdaInst.asInstanceOf[(FeatureType, FeatureType, FeatureType) => FeatureType]
                )(tti1, tti2, tti3, tto, ttov)
              }
              // scalastyle:off
              case ClassQuaternaryLambdaTransformerName => {
                val tti2 = getFeatureTypeTag(metadataJson, FieldNames.LambdaTypeI2)
                val tti3 = getFeatureTypeTag(metadataJson, FieldNames.LambdaTypeI3)
                val tti4 = getFeatureTypeTag(metadataJson, FieldNames.LambdaTypeI4)
                new QuaternaryLambdaTransformer[FeatureType, FeatureType, FeatureType, FeatureType, FeatureType](
                  uid = uid,
                  operationName = opName,
                  transformFn = lambdaInst.asInstanceOf[(FeatureType, FeatureType, FeatureType, FeatureType) => FeatureType]
                )(tti1, tti2, tti3, tti4, tto, ttov)
              }
              // scalastyle:on

              case _ => throw new IllegalArgumentException(s"Don't know how to instantinate ${name}")
            }
        }

    }

  }

  /**
   * Loads from the json serialized data
   *
   * @param jsonStr     json string
   * @param path        to the stored output
   * @param loadLambdas try to load lambdas from json rather than from workflow
   * @return OpPipelineStageBase
   */
  def loadFromJsonString(jsonStr: String, path: String, loadLambdas: Boolean = false): OpPipelineStageBase = {
    // Load stage json with it's params
    val metadata = SparkDefaultParamsReadWrite.parseMetadata(jsonStr)
    val (className, metadataJson) = metadata.className -> metadata.metadata
    // Check if it's a model
    val isModel = (metadataJson \ FieldNames.IsModel.entryName).extract[Boolean]
    val hasCtorAgrs = (metadataJson \ FieldNames.CtorArgs.entryName) != JNothing

    // In case we stumbled upon model we instantiate it using the class name + ctor args
    // otherwise we simply use the provided stage instance.
    // UnaryTransformer

    val stage = loadLambdaTransformer(metadataJson, className) match {
      case Some(Success(s)) => s
      case Some(Failure(e)) => throw e
      case _ if isModel || hasCtorAgrs => loadModel(className, metadataJson)
      case _: LambdaTransformer[_, _] if loadLambdas =>
        throw new RuntimeException("Loaded transformer stage from workflow instead of from JSON!")
      case _ => originalStage
    }

    // Recover all stage spark params and it's input features
    val inputFeatures = originalStage.getInputFeatures()

    // Update [[SparkWrapperParams]] with path so we can load the [[SparkStageParam]] instance
    val updatedMetadata = stage match {
      case _: SparkWrapperParams[_] =>
        val updatedParams = SparkStageParam.updateParamsMetadataWithPath(metadata.params, path)
        metadata.copy(params = updatedParams)
      case _ => metadata
    }

    // Set all stage params from the metadata
    SparkDefaultParamsReadWrite.getAndSetParams(stage, updatedMetadata)
    val matchingFeatures = stage.getTransientFeatures().map { f =>
      inputFeatures.find(i => i.uid == f.uid && i.isResponse == f.isResponse && i.typeName == f.typeName)
        .getOrElse(throw new RuntimeException(s"Feature '${f.uid}' was not found for stage '${stage.uid}'"))
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
    val ctorArgsValues = ctorArgsJson.map { case (argName, j) => argName -> j \ "value" }.toMap
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

        case AnyValueTypes.Class => ReflectionUtils.newLambdaInstance(anyValue.value.toString) match {
          case Success(ci) => ci
          case Failure(e) => throw e
        }

        // Spark wrapped stage is saved using [[SparkWrapperParams]] and loaded later using
        // [[SparkDefaultParamsReadWrite]].getAndSetParams returning 'null' here
        case AnyValueTypes.SparkWrappedStage => {
          null // yes, yes - this should be 'null'
        }


        // Everything else is read using json4s
        case AnyValueTypes.Value => Try {
          ctorArgsValues(argName) match {
            case a: JArray => {
              val v = jsonToVal(a.arr.toArray)
              v
            }
            case _ =>
              val ttag = ReflectionUtils.typeTagForType[Any](tpe = argSymbol.info)
              val manifest = ReflectionUtils.manifestForTypeTag[Any](ttag)
              Extraction.decompose(anyValue.value).extract[Any](formats, manifest)
          }
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
