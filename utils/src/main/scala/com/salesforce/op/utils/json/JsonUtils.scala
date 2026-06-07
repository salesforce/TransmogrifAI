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

package com.salesforce.op.utils.json


import java.io.File
import com.fasterxml.jackson.annotation.JsonAutoDetect.Visibility
import com.fasterxml.jackson.annotation.JsonInclude.Include
import com.fasterxml.jackson.annotation.PropertyAccessor
import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind._
import com.fasterxml.jackson.databind.module.SimpleModule
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.commons.io.FilenameUtils

import scala.reflect._
import scala.util.Try

/**
 * Json/Yaml marshalling utils
 */
object JsonUtils {

  /**
   * Read object from a json or yaml file
   *
   * @param file   json or yaml file
   * @param serdes custom serializers
   * @return Try[T]
   */
  def fromFile[T: ClassTag](file: File, serdes: Seq[SerDes[_]] = Seq.empty): Try[T] = Try {
    val extension = FilenameUtils.getExtension(file.getPath).toLowerCase
    val mapper: ObjectMapper =
      extension match {
        case "json" => jsonMapper(serdes)
        case y if "yml" == y || "yaml" == y => yamlMapper(serdes)
        case _ => throw new IllegalArgumentException(
          s"Unsupported file type '$extension'. Supported file types: json, yml, yaml")
      }
    mapper.readValue(file, classTag[T].runtimeClass).asInstanceOf[T]
  }

  /**
   * Read object from a json or yaml string
   *
   * @param str    json or yaml string
   * @param serdes custom serializers/deserializers
   * @return Try[T]
   */
  def fromString[T: ClassTag](str: String, serdes: Seq[SerDes[_]] = Seq.empty): Try[T] = Try {
    jsonMapper(serdes).readValue(str, classTag[T].runtimeClass)
  }.recover { case _ =>
    yamlMapper(serdes).readValue(str, classTag[T].runtimeClass)
  }.recover { case e =>
    throw new IllegalArgumentException("Unsupported format. Supported formats: json, yaml", e)
  }.map(_.asInstanceOf[T])

  /**
   * Write an instance to json string
   *
   * @param any    instance
   * @param pretty should pretty print
   * @param serdes custom serializers/deserializers
   * @return json string of the instance
   */
  def toJsonString(any: AnyRef, pretty: Boolean = true, serdes: Seq[SerDes[_]] = Seq.empty): String = {
    val writer = if (pretty) jsonMapper(serdes).writerWithDefaultPrettyPrinter() else jsonMapper(serdes).writer()
    writer.writeValueAsString(any)
  }

  /**
   * Write an instance to json node
   *
   * @param any    instance
   * @param serdes custom serializers/deserializers
   * @return root node of the resulting json tree
   */
  def toJsonTree(any: AnyRef, serdes: Seq[SerDes[_]] = Seq.empty): JsonNode = jsonMapper(serdes).valueToTree(any)

  /**
   * Write json node into a Map
   *
   * @param json   json node
   * @param serdes custom serializers/deserializers
   * @return Map
   */
  def toMap(json: JsonNode, serdes: Seq[SerDes[_]] = Seq.empty): Map[String, Any] =
    jsonMapper(serdes).convertValue(json, Map.empty[String, Any].getClass)

  /**
   * Write an instance to yaml string
   *
   * @param any    instance
   * @param pretty should pretty print
   * @param serdes custom serializers/deserializers
   * @return yaml string of the instance
   */
  def toYamlString(any: AnyRef, pretty: Boolean = true, serdes: Seq[SerDes[_]] = Seq.empty): String = {
    val writer = if (pretty) yamlMapper(serdes).writerWithDefaultPrettyPrinter() else jsonMapper(serdes).writer()
    writer.writeValueAsString(any)
  }

  private def jsonMapper(serdes: Seq[SerDes[_]]): ObjectMapper = configureMapper(serdes) {
    new ObjectMapper()
      .configure(JsonParser.Feature.ALLOW_COMMENTS, true)
      .configure(JsonParser.Feature.ALLOW_SINGLE_QUOTES, true)
      .configure(JsonParser.Feature.ALLOW_UNQUOTED_FIELD_NAMES, true)
      .registerModule(DefaultScalaModule)
  }

  private def yamlMapper(serdes: Seq[SerDes[_]]): ObjectMapper = configureMapper(serdes) {
    new ObjectMapper(new YAMLFactory())
      .configure(JsonParser.Feature.ALLOW_YAML_COMMENTS, true)
  }

  private def configureMapper(serdes: Seq[SerDes[_]])(mapper: ObjectMapper): ObjectMapper = {
    if (serdes.nonEmpty) {
      val extraSerdes = new SimpleModule()
      serdes.foreach { serde =>
        extraSerdes
          .addSerializer(serde.ser)
          .addDeserializer(serde.klazz.asInstanceOf[Class[Any]], serde.des)
      }
      mapper.registerModule(extraSerdes)
    }
    mapper
      .setSerializationInclusion(Include.NON_NULL)
      .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
      .setVisibility(PropertyAccessor.FIELD, Visibility.ANY)
      .registerModule(DefaultScalaModule)
  }

}

/**
 * Custom serializer/deserializer container
 */
case class SerDes[T](klazz: Class[T], ser: JsonSerializer[T], des: JsonDeserializer[T])

/**
 * To/from Json mixin
 */
trait JsonLike extends Serializable {
  /**
   * Write this instance to json string
   *
   * @param pretty should pretty print
   * @return json string of the instance
   */
  def toJson(pretty: Boolean = true): String = JsonUtils.toJsonString(this, pretty = pretty)

  /**
   * This instance json string
   *
   * @return json string of the instance
   */
  override def toString: String = this.toJson(pretty = true)
}
