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
import com.salesforce.op.stages.OpPipelineStageReadWriteShared._
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.spark.ml.PipelineStage
import org.json4s.{JObject, JValue}
import org.json4s.jackson.JsonMethods.render
import org.json4s.{Extraction, _}

import scala.reflect.{ClassTag, ManifestFactory}
import scala.reflect.runtime.universe._
import scala.util.{Failure, Success, Try}

/**
 * Default reader/writer for stages that uses reflection to reflect stage ctor arguments
 *
 * @tparam StageType stage type to read/write
 */
final class DefaultOpPipelineStageJsonReaderWriter[StageType <: OpPipelineStageBase]
(
  implicit val ct: ClassTag[StageType]
) extends OpPipelineStageJsonReaderWriter[StageType] with SerializationFuns {

  /**
   * Read stage from json
   *
   * @param stageClass stage class
   * @param json       json to read stage from
   * @return read result
   */
  def read(stageClass: Class[StageType], json: JValue): Try[StageType] = Try {
    // Extract all the ctor args
    val ctorArgsMap = json.asInstanceOf[JObject].obj
      .map { case (argName, j) => argName -> j.extract[AnyValue] }.toMap

    // Make the ctor function used for creating a stage instance
    val ctorArgs: (String, Symbol) => Try[Any] = (argName, argSymbol) => {
      for {
        anyValue <- Try {
          ctorArgsMap.getOrElse(argName, throw new RuntimeException(
            s"Ctor argument '$argName' was not found for stage class '${stageClass.getName}'"))
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
              ReflectionUtils.newInstance[Any](value.toString)

            // Value with no ctor arguments should be instantiable by class name
            case AnyValue(AnyValueTypes.Value, m: Map[_, _], Some(className)) if m.isEmpty =>
              ReflectionUtils.newInstance[Any](className)

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
    ReflectionUtils.newInstance[StageType](stageClass, ctorArgs)
  }

  /**
   * Write stage to json
   *
   * @param stage stage instance to write
   * @return write result
   */
  def write(stage: StageType): Try[JValue] = Try {
    // Reflect all the ctor args
    val (_, argsList) = ReflectionUtils.bestCtorWithArgs(stage)

    // Wrap all ctor args into AnyValue container
    val args =
      for {(argName, argValue) <- argsList} yield {
        val anyValue = argValue match {

          // Special handling for Feature Type TypeTags
          case t: TypeTag[_] if FeatureType.isFeatureType(t) || FeatureType.isFeatureValueType(t) =>
            AnyValue(AnyValueTypes.TypeTag, ReflectionUtils.dealisedTypeName(t.tpe), None)
          case t: TypeTag[_] =>
            throw new RuntimeException(
              s"Unknown type tag '${t.tpe.toString}'. " +
                "Only Feature and Feature Value type tags are supported for serialization."
            )

          // Special handling for function value arguments
          case f1: Function1[_, _]
            // Maps and other scala collections extend [[Function1]] - skipping them by filtering by package name
            if !f1.getClass.getPackage.getName.startsWith("scala") => serializeArgument(argName, f1)
          case f2: Function2[_, _, _] => serializeArgument(argName, f2)
          case f3: Function3[_, _, _, _] => serializeArgument(argName, f3)
          case f4: Function4[_, _, _, _, _] => serializeArgument(argName, f4)

          // Special handling for [[Numeric]]
          case n: Numeric[_] => serializeArgument(argName, n)

          // Spark wrapped stage is saved using [[SparkWrapperParams]], so we just writing it's uid here
          case Some(v: PipelineStage) => AnyValue(AnyValueTypes.SparkWrappedStage, v.uid, None)
          case v: PipelineStage => AnyValue(AnyValueTypes.SparkWrappedStage, v.uid, None)

          // Everything else goes as is and is handled by json4s
          case v =>
            // try serialize value with json4s
            val av = AnyValue(AnyValueTypes.Value, v, Option(v).map(_.getClass.getName))
            Try(jsonSerialize(av)) match {
              case Success(_) => av
              case Failure(e) =>
                throw new RuntimeException(s"Failed to json4s serialize argument '$argName' with value '$v'", e)
            }

        }
        argName -> anyValue
      }
    Extraction.decompose(args.toMap)
  }


  private def jsonSerialize(v: Any): JValue = render(Extraction.decompose(v))
}
