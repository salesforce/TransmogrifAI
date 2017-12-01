/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.sql.types

import org.json4s.JValue
import org.json4s.jackson.JsonMethods.{pretty, render}

import scala.reflect.ClassTag
import scala.util.Try

/**
 * Metadata wrapper allows access to private members of metadata
 *
 * @param metadata a Metadata instance
 */
class MetadataWrapper(val metadata: Metadata) {

  /**
   * Unsafe ctor to allow creation of Metadata using untyped map.
   * Note: use this ctor only as a last resort solution, otherwise always prefer [[MetadataBuilder]].
   *
   * @param m untyped map
   * @return [[Metadata]]
   */
  def this(m: Map[String, Any]) = this(new Metadata(m))

  /**
   * Get underlying untyped Map[String, Any].
   * Since Metadata.map val is private, this is the only way for us to get it
   *
   * @return underlying untyped Map[String, Any]
   */
  def underlyingMap: Map[String, Any] = metadata.map

  /**
   * Optionally returns the value associated with a key.
   *
   * @param  key the key value
   * @return an option value containing the value associated with `key` in this map,
   *         or `None` if none exists.
   */
  def getAny(key: String): Option[Any] = underlyingMap.get(key)

  /**
   * Returns the value associated with a key.
   *
   * @param  key the key value
   * @return  value containing the value associated with `key` in this map cast to the type provided
   */
  def get[T](key: String): T = underlyingMap(key).asInstanceOf[T]


  /**
   * Returns the value associated with a key. Handles empty arrays serialized to Long instead of proper type.
   *
   * @param  key the key value for an array
   * @return  value containing the value associated with `key` in this map cast to the array type provided
   */
  def getArray[T : ClassTag](key: String): Array[T] = {
    underlyingMap(key) match {
      case ar: Array[_] if ar.isEmpty => Array.empty[T] // Spark metadata changes type to Long when empty is serialized
      case ar: Array[_] => ar.map(_.asInstanceOf[T])
      case other => throw new ClassCastException(
        s"Metadata $other cannot be cast to type ${classOf[Array[T]]}"
      )
    }
  }

  /**
   * Retrieves the value which is associated with the given key. This
   * method invokes the `default` method of the map if there is no mapping
   * from the given key to a value. Unless overridden, the `default` method throws a
   * `NoSuchElementException`.
   *
   * @param  key the key
   * @return the value associated with the given key, or the result of the
   *         map's `default` method, if none exists.
   */
  def apply(key: String): Any = underlyingMap(key)

  /**
   * Tests whether this map contains a binding for a key.
   *
   * @param key the key
   * @return `true` if there is a binding for `key` in this map, `false` otherwise.
   */
  def contains(key: String): Boolean = metadata.contains(key)

  /**
   * Converts to its pretty json representation
   *
   * @return pretty json string
   */
  def prettyJson: String = pretty(render(jsonValue))

  /**
   * Metadata Json Value
   * @return Json Value
   */
  def jsonValue: JValue = metadata.jsonValue

}
