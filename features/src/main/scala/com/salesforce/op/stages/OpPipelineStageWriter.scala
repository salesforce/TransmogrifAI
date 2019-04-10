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
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.hadoop.fs.Path
import OpPipelineStageReadWriteShared._
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import org.apache.spark.ml.util.MLWriter
import org.apache.spark.ml.{Model, PipelineStage, SparkDefaultParamsReadWrite}
import org.json4s.Extraction
import org.json4s.JsonAST.{JObject, JValue}
import org.json4s.jackson.JsonMethods.{compact, parse, render}

import scala.reflect.runtime.universe.TypeTag
import scala.util.{Failure, Success, Try}

/**
 * MLWriter class used to write TransmogrifAI stages to disk
 *
 * @param stage a stage to save
 */
final class OpPipelineStageWriter(val stage: OpPipelineStageBase) extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    val metadataPath = new Path(path, "metadata").toString
    sc.parallelize(Seq(writeToJsonString(path)), 1).saveAsTextFile(metadataPath)
  }

  /**
   * Stage metadata json string
   *
   * @return stage metadata json string
   */
  def writeToJsonString(path: String): String = compact(writeToJson(path))

  /**
   * Stage metadata json
   *
   * @return stage metadata json
   */
  def writeToJson(path: String): JObject = jsonSerialize(writeToMap(path)).asInstanceOf[JObject]

  /**
   * Stage metadata map
   *
   * @return stage metadata map
   */
  def writeToMap(path: String): Map[String, Any] = {
    // Set save path for all Spark wrapped stages of type [[SparkWrapperParams]]
    // so they can save
    stage match {
      case s: SparkWrapperParams[_] => s.setStageSavePath(path)
      case s => s
    }
    // We produce stage metadata for all the Spark params
    val metadataJson = SparkDefaultParamsReadWrite.getMetadataToSave(stage)
    val metadata = parse(metadataJson).extract[Map[String, Any]]

    // We also include stage it's ctor args, so we can reconstruct the model instance when loading
    metadata + (FieldNames.CtorArgs.entryName -> modelCtorArgs().toMap)
  }

  /**
   * Extract model ctor args values keyed by their names, so we can reconstruct the model instance when loading.
   * See [[OpPipelineStageReader]].OpPipelineStageReader
   *
   * @return model ctor args values by their names
   */
  private def modelCtorArgs(): Seq[(String, AnyValue)] = Try {
    // Reflect all model ctor args
    val (_, argsList) = ReflectionUtils.bestCtorWithArgs(stage)

    // Wrap all ctor args into AnyValue container
    for {(argName, argValue) <- argsList} yield {
      val anyValue = argValue match {

        // Special handling for Feature Type TypeTags
        case t: TypeTag[_] if FeatureType.isFeatureType(t) || FeatureType.isFeatureValueType(t) =>
          new AnyValue(AnyValueTypes.TypeTag, ReflectionUtils.dealisedTypeName(t.tpe))
        case t: TypeTag[_] =>
          throw new RuntimeException(
            s"Unknown type tag '${t.tpe.toString}'. " +
              "Only Feature and Feature Value type tags are supported for serialization."
          )

        // Special handling for function value arguments
        case f1: Function1[_, _]
          // Maps and other scala collections extend [[Function1]] - skipping them
          if !f1.getClass.getPackage.getName.startsWith("scala") => serializeFunction(argName, f1)
        case f2: Function2[_, _, _] => serializeFunction(argName, f2)
        case f3: Function3[_, _, _, _] => serializeFunction(argName, f3)
        case f4: Function4[_, _, _, _, _] => serializeFunction(argName, f4)

        // Special handling for [[Numeric]]
        case n: Numeric[_] => new AnyValue(AnyValueTypes.ClassInstance, n.getClass.getName, n.getClass.getName)

        // Spark wrapped stage is saved using [[SparkWrapperParams]], so we just writing it's uid here
        case Some(v: PipelineStage) => new AnyValue(AnyValueTypes.SparkWrappedStage, v.uid)
        case v: PipelineStage => new AnyValue(AnyValueTypes.SparkWrappedStage, v.uid)

        // Everything else goes as is and is handled by json4s
        case v =>
          // try serialize value with json4s
          val av = new AnyValue(AnyValueTypes.Value, v)
          Try(jsonSerialize(av)) match {
            case Success(_) => av
            case Failure(e) =>
              throw new RuntimeException(s"Failed to json4s serialize argument '$argName' with value '$v'", e)
          }

      }
      argName -> anyValue
    }
  } match {
    case Success(args) => args
    case Failure(error) =>
      throw new RuntimeException(s"Failed to extract constructor arguments for model stage '${stage.uid}'. " +
        "Make sure your model class is a concrete final class with json4s serializable arguments.", error
      )
  }

  private def serializeFunction(argName: String, function: AnyRef): AnyValue = {
    val functionClass = function.getClass
    try {
      // Test that function has no external dependencies and can be constructed without ctor args
      functionClass.getConstructors.head.newInstance()
    } catch {
      case e: Exception => throw new RuntimeException(
        s"Function argument '$argName' cannot be serialized. " +
          "Make sure your function does not have any external dependencies, " +
          "e.g. use any out of scope variables.", e)
    }
    new AnyValue(
      `type` = AnyValueTypes.ClassInstance, value = functionClass.getName, valueClass = functionClass.getName)
  }

  private def jsonSerialize(v: Any): JValue = render(Extraction.decompose(v))
}
