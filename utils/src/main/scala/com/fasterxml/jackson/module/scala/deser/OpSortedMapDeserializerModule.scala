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
import com.fasterxml.jackson.module.scala.introspect.OrderingLocator
import com.fasterxml.jackson.module.scala.modifiers.MapTypeModifierModule

import scala.collection.immutable.TreeMap
import scala.collection.{SortedMap, mutable}
import scala.language.existentials


// scalastyle:off
private class SortedMapBuilderWrapper[K, V](val builder: mutable.Builder[(K, V), SortedMap[K, V]]) extends AbstractMap[K, V] {
  override def put(k: K, v: V) = {
    builder += ((k, v));
    v
  }

  // Isn't used by the deserializer
  def entrySet(): java.util.Set[Entry[K, V]] = throw new UnsupportedOperationException
}

private object SortedMapDeserializer {
  def orderingFor = OrderingLocator.locate _

  def builderFor(cls: Class[_], keyCls: JavaType): mutable.Builder[(AnyRef, AnyRef), SortedMap[AnyRef, AnyRef]] =
    if (classOf[TreeMap[_, _]].isAssignableFrom(cls)) TreeMap.newBuilder[AnyRef, AnyRef](orderingFor(keyCls)) else
      SortedMap.newBuilder[AnyRef, AnyRef](orderingFor(keyCls))
}

private class SortedMapDeserializer(
  collectionType: MapLikeType,
  config: DeserializationConfig,
  keyDeser: KeyDeserializer,
  valueDeser: JsonDeserializer[_],
  valueTypeDeser: TypeDeserializer)
  extends ContainerDeserializerBase[SortedMap[_, _]](collectionType)
    with ContextualDeserializer {

  private val javaContainerType =
    config.getTypeFactory.constructMapLikeType(classOf[MapBuilderWrapper[_, _]], collectionType.containedType(0), collectionType.containedType(1))

  private val instantiator =
    new ValueInstantiator {
      def getValueTypeDesc = collectionType.getRawClass.getCanonicalName

      override def canCreateUsingDefault = true

      override def createUsingDefault(ctx: DeserializationContext) =
        new SortedMapBuilderWrapper[AnyRef, AnyRef](SortedMapDeserializer.builderFor(collectionType.getRawClass, collectionType.containedType(0)))
    }

  private val containerDeserializer =
    new MapDeserializer(javaContainerType, instantiator, keyDeser, valueDeser.asInstanceOf[JsonDeserializer[AnyRef]], valueTypeDeser)

  override def getContentType = containerDeserializer.getContentType

  override def getContentDeserializer = containerDeserializer.getContentDeserializer

  override def createContextual(ctxt: DeserializationContext, property: BeanProperty) =
    if (keyDeser != null && valueDeser != null) this
    else {
      val newKeyDeser = Option(keyDeser).getOrElse(ctxt.findKeyDeserializer(collectionType.getKeyType, property))
      val newValDeser = Option(valueDeser).getOrElse(ctxt.findContextualValueDeserializer(collectionType.getContentType, property))
      new SortedMapDeserializer(collectionType, config, newKeyDeser, newValDeser, valueTypeDeser)
    }

  override def deserialize(jp: JsonParser, ctxt: DeserializationContext): SortedMap[_, _] = {
    containerDeserializer.deserialize(jp, ctxt) match {
      case wrapper: SortedMapBuilderWrapper[_, _] => wrapper.builder.result()
    }
  }

  //**********************************************************************************
  /**
   * In order to allow deserialization of nulls as empty maps we override this method.
   * The fix is inspired by - https://github.com/FasterXML/jackson-module-scala/pull/257
   */
  //**********************************************************************************
  override def getNullValue(ctxt: DeserializationContext): SortedMap[_, _] = {
    SortedMapDeserializer.builderFor(collectionType.getRawClass, collectionType.containedType(0)).result()
  }
}

private object SortedMapDeserializerResolver extends Deserializers.Base {

  private val SORTED_MAP = classOf[collection.SortedMap[_, _]]

  override def findMapLikeDeserializer(theType: MapLikeType,
    config: DeserializationConfig,
    beanDesc: BeanDescription,
    keyDeserializer: KeyDeserializer,
    elementTypeDeserializer: TypeDeserializer,
    elementDeserializer: JsonDeserializer[_]): JsonDeserializer[_] =
    if (!SORTED_MAP.isAssignableFrom(theType.getRawClass)) null
    else new SortedMapDeserializer(theType, config, keyDeserializer, elementDeserializer, elementTypeDeserializer)

}

/**
 * @author Christopher Currie <christopher@currie.com>
 */
trait OpSortedMapDeserializerModule extends MapTypeModifierModule {
  this += (_ addDeserializers SortedMapDeserializerResolver)
}
