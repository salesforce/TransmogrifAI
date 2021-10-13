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

package com.fasterxml.jackson.module.scala

import com.fasterxml.jackson.module.scala.deser._
import com.fasterxml.jackson.module.scala.introspect.ScalaAnnotationIntrospectorModule
import com.fasterxml.jackson.module.scala.ser.MapSerializerModule

// scalastyle:off
class OpDefaultScalaModule
  extends JacksonModule
    with IteratorModule
    with EnumerationModule
    with OptionModule
    with SeqModule
    with IterableModule
    with TupleModule
    //**********************************************************************************
    /**
     * In order to allow deserialization of nulls as empty maps.
     * replaced [[UnsortedMapDeserializerModule]] and [[SortedMapDeserializerModule]]
     * with [[OpUnsortedMapDeserializerModule]] and [[OpSortedMapDeserializerModule]]
     * respectively.
     *
     * The fix is inspired by - https://github.com/FasterXML/jackson-module-scala/pull/257
     */
    //**********************************************************************************
    with OpUnsortedMapDeserializerModule
    with OpSortedMapDeserializerModule
    with MapSerializerModule
    //**********************************************************************************
    with SetModule
    with ScalaNumberDeserializersModule
    with ScalaAnnotationIntrospectorModule
    with UntypedObjectDeserializerModule
    with EitherModule
{
  override def getModuleName = "OpDefaultScalaModule"
}

object OpDefaultScalaModule extends OpDefaultScalaModule
