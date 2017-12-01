/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.json


import java.io.File

import com.fasterxml.jackson.annotation.JsonAutoDetect.Visibility
import com.fasterxml.jackson.annotation.JsonInclude.Include
import com.fasterxml.jackson.annotation.PropertyAccessor
import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind._
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.OpDefaultScalaModule
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
   * @param file json or yaml file
   * @return Try[T]
   */
  def fromFile[T: ClassTag](file: File): Try[T] = Try {
    val extension = FilenameUtils.getExtension(file.getPath).toLowerCase
    val mapper: ObjectMapper =
      extension match {
        case "json" => jsonMapper
        case y if "yml" == y || "yaml" == y => yamlMapper
        case _ => throw new IllegalArgumentException(
          s"Unsupported file type '$extension'. Supported file types: json, yml, yaml")
      }
    mapper.readValue(file, classTag[T].runtimeClass).asInstanceOf[T]
  }

  /**
   * Read object from a json or yaml string
   *
   * @param str json or yaml string
   * @return Try[T]
   */
  def fromString[T: ClassTag](str: String): Try[T] = Try {
    jsonMapper.readValue(str, classTag[T].runtimeClass)
  }.recover { case _ =>
    yamlMapper.readValue(str, classTag[T].runtimeClass)
  }.recover { case e =>
    throw new IllegalArgumentException("Unsupported format. Supported formats: json, yaml", e)
  }.map(_.asInstanceOf[T])

  /**
   * Write an instance to json string
   *
   * @param any    instance
   * @param pretty should pretty print
   * @return json string of the instance
   */
  def toJsonString(any: AnyRef, pretty: Boolean = true): String = {
    val writer = if (pretty) jsonMapper.writerWithDefaultPrettyPrinter() else jsonMapper.writer()
    writer.writeValueAsString(any)
  }

  /**
   * Write an instance to json node
   *
   * @param any instance
   * @return root node of the resulting json tree
   */
  def toJsonTree(any: AnyRef): JsonNode = jsonMapper.valueToTree(any)

  /**
   * Write json node into a Map
   *
   * @param json json node
   * @return Map
   */
  def toMap(json: JsonNode): Map[String, Any] = jsonMapper.convertValue(json, Map.empty[String, Any].getClass)

  /**
   * Write an instance to yaml string
   *
   * @param any    instance
   * @param pretty should pretty print
   * @return yaml string of the instance
   */
  def toYamlString(any: AnyRef, pretty: Boolean = true): String = {
    val writer = if (pretty) yamlMapper.writerWithDefaultPrettyPrinter() else jsonMapper.writer()
    writer.writeValueAsString(any)
  }

  private def jsonMapper: ObjectMapper = configureMapper {
    new ObjectMapper()
      .configure(JsonParser.Feature.ALLOW_COMMENTS, true)
      .configure(JsonParser.Feature.ALLOW_SINGLE_QUOTES, true)
      .configure(JsonParser.Feature.ALLOW_UNQUOTED_FIELD_NAMES, true)
  }

  private def yamlMapper: ObjectMapper = configureMapper {
    new ObjectMapper(new YAMLFactory())
      .configure(JsonParser.Feature.ALLOW_YAML_COMMENTS, true)
  }

  private def configureMapper(mapper: ObjectMapper): ObjectMapper = {
    mapper
      .setSerializationInclusion(Include.NON_NULL)
      .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
      .setVisibility(PropertyAccessor.FIELD, Visibility.ANY)
      .registerModule(OpDefaultScalaModule)
  }

}

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
