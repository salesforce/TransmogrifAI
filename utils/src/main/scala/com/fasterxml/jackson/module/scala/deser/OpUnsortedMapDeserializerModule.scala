// scalastyle:off header.matches
/**
 * Modifications: (c) 2017, Salesforce.com, Inc.
 * Copyright 2017 Fasterxml.com
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.fasterxml.jackson.module.scala.deser

import java.util.AbstractMap
import java.util.Map.Entry

import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind._
import com.fasterxml.jackson.databind.`type`.MapLikeType
import com.fasterxml.jackson.databind.deser.std.{ContainerDeserializerBase, MapDeserializer}
import com.fasterxml.jackson.databind.deser.{ContextualDeserializer, Deserializers, ValueInstantiator}
import com.fasterxml.jackson.databind.jsontype.TypeDeserializer
import com.fasterxml.jackson.module.scala.modifiers.MapTypeModifierModule

import scala.collection.immutable.HashMap
import scala.collection.{GenMap, mutable}
import scala.language.existentials


// scalastyle:off
private class MapBuilderWrapper[K,V](val builder: mutable.Builder[(K,V), GenMap[K,V]]) extends AbstractMap[K,V] {
  override def put(k: K, v: V) = { builder += ((k,v)); v }

  // Isn't used by the deserializer
  def entrySet(): java.util.Set[Entry[K, V]] = throw new UnsupportedOperationException
}

private object UnsortedMapDeserializer {
  def builderFor(cls: Class[_]): mutable.Builder[(AnyRef,AnyRef), GenMap[AnyRef,AnyRef]] =
    if (classOf[HashMap[_,_]].isAssignableFrom(cls)) HashMap.newBuilder[AnyRef,AnyRef] else
    if (classOf[mutable.HashMap[_,_]].isAssignableFrom(cls)) mutable.HashMap.newBuilder[AnyRef,AnyRef] else
    if (classOf[mutable.ListMap[_,_]].isAssignableFrom(cls)) mutable.ListMap.newBuilder[AnyRef,AnyRef] else
    if (classOf[mutable.LinkedHashMap[_,_]].isAssignableFrom(cls)) mutable.LinkedHashMap.newBuilder[AnyRef,AnyRef] else
    if (classOf[mutable.Map[_,_]].isAssignableFrom(cls)) mutable.Map.newBuilder[AnyRef,AnyRef]
    else Map.newBuilder[AnyRef,AnyRef]
}

private class UnsortedMapDeserializer(
  collectionType: MapLikeType,
  config: DeserializationConfig,
  keyDeser: KeyDeserializer,
  valueDeser: JsonDeserializer[_],
  valueTypeDeser: TypeDeserializer)

  extends ContainerDeserializerBase[GenMap[_,_]](config.constructType(classOf[UnsortedMapDeserializer]))
    with ContextualDeserializer {

  private val javaContainerType =
    config.getTypeFactory.constructMapLikeType(classOf[MapBuilderWrapper[_,_]], collectionType.containedType(0), collectionType.containedType(1))

  private val instantiator =
    new ValueInstantiator {
      override def getValueTypeDesc = collectionType.getRawClass.getCanonicalName
      override def canCreateUsingDefault = true
      override def createUsingDefault(ctxt: DeserializationContext) =
        new MapBuilderWrapper[AnyRef,AnyRef](UnsortedMapDeserializer.builderFor(collectionType.getRawClass))
    }

  private val containerDeserializer =
    new MapDeserializer(javaContainerType,instantiator,keyDeser,valueDeser.asInstanceOf[JsonDeserializer[AnyRef]],valueTypeDeser)

  override def getContentType = containerDeserializer.getContentType

  override def getContentDeserializer = containerDeserializer.getContentDeserializer

  override def createContextual(ctxt: DeserializationContext, property: BeanProperty) =
    if (keyDeser != null && valueDeser != null) this
    else {
      val newKeyDeser = Option(keyDeser).getOrElse(ctxt.findKeyDeserializer(collectionType.getKeyType, property))
      val newValDeser = Option(valueDeser).getOrElse(ctxt.findContextualValueDeserializer(collectionType.getContentType, property))
      new UnsortedMapDeserializer(collectionType, config, newKeyDeser, newValDeser, valueTypeDeser)
    }

  override def deserialize(jp: JsonParser, ctxt: DeserializationContext): GenMap[_,_] = {
    containerDeserializer.deserialize(jp,ctxt) match {
      case wrapper: MapBuilderWrapper[_,_] => wrapper.builder.result()
    }
  }

  //**********************************************************************************
  /**
   * In order to allow deserialization of nulls as empty maps we override this method.
   * The fix is inspired by - https://github.com/FasterXML/jackson-module-scala/pull/257
   */
  //**********************************************************************************
  override def getNullValue(ctxt: DeserializationContext): GenMap[_, _] = {
    UnsortedMapDeserializer.builderFor(collectionType.getRawClass).result()
  }
}

private object UnsortedMapDeserializerResolver extends Deserializers.Base {

  private val MAP = classOf[collection.Map[_,_]]
  private val SORTED_MAP = classOf[collection.SortedMap[_,_]]

  override def findMapLikeDeserializer(theType: MapLikeType,
    config: DeserializationConfig,
    beanDesc: BeanDescription,
    keyDeserializer: KeyDeserializer,
    elementTypeDeserializer: TypeDeserializer,
    elementDeserializer: JsonDeserializer[_]): JsonDeserializer[_] = {
    val rawClass = theType.getRawClass

    if (!MAP.isAssignableFrom(rawClass)) null
    else if (SORTED_MAP.isAssignableFrom(rawClass)) null
    else new UnsortedMapDeserializer(theType, config, keyDeserializer, elementDeserializer, elementTypeDeserializer)
  }

}

/**
 * @author Christopher Currie <ccurrie@impresys.com>
 */
trait OpUnsortedMapDeserializerModule extends MapTypeModifierModule {
  this += { _.addDeserializers(UnsortedMapDeserializerResolver) }
}
