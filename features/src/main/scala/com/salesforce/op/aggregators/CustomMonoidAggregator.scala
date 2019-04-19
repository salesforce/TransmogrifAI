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

package com.salesforce.op.aggregators

import com.salesforce.op.features.types.{FeatureType, FeatureTypeFactory, Text}
import com.salesforce.op.stages.base.binary.BinaryTransformer
import com.salesforce.op.stages.{OpPipelineStageJsonReaderWriter, ReaderWriter}
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.json4s.JsonDSL._
import com.twitter.algebird.{Monoid, MonoidAggregator}
import org.json4s.jackson.JsonMethods.render
import org.json4s.{Extraction, JValue}

import scala.reflect.runtime.universe.TypeTag
import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Try

/**
 * Custom Monoid Aggregator allowing passing a zero value and an associative function to combine values
 *
 * @param zero          zero value
 * @param associativeFn associative function to combine values
 * @tparam O type of feature
 */
@ReaderWriter(classOf[CustomMonoidAggregatorReaderWriter[_ <: FeatureType]])
case class CustomMonoidAggregator[O <: FeatureType]
(
  zero: O#Value,
  associativeFn: (O#Value, O#Value) => O#Value
)(implicit val ttag: WeakTypeTag[O], val ttov: WeakTypeTag[O#Value])
  extends MonoidAggregator[Event[O], O#Value, O] with AggregatorDefaults[O] {
  val ftFactory = FeatureTypeFactory[O]()
  val monoid: Monoid[O#Value] = Monoid.from(zero)(associativeFn)
}


class CustomMonoidAggregatorReaderWriter[T <: FeatureType]
  extends OpPipelineStageJsonReaderWriter[CustomMonoidAggregator[T]] {
  /**
   * Read stage from json
   *
   * @param stageClass stage class
   * @param json       json to read stage from
   * @return read result
   */
  override def read(stageClass: Class[CustomMonoidAggregator[T]], json: JValue): Try[CustomMonoidAggregator[T]] = {
    Try {
      val tto = FeatureType.featureTypeTag((json \ "tto").extract[String]).asInstanceOf[TypeTag[T]]

      val ttov = FeatureType.featureTypeTag((json \ "ttov").extract[String]).asInstanceOf[TypeTag[T#Value]]
      val fnc = ReflectionUtils.classForName((json \ "fn").extract[String]).getConstructors.head.newInstance()
        .asInstanceOf[Function2[T#Value, T#Value, T#Value]]
      val manifest = ReflectionUtils.manifestForTypeTag[T#Value](ttov)
      val zero = Extraction.decompose(json \ "zero").extract[T#Value](formats, manifest)
      CustomMonoidAggregator(zero, fnc)(tto, ttov)
    }
    /*

     case AnyValue(AnyValueTypes.Value, value, valueClass) =>
              // Create type manifest either using the reflected type tag or serialized value class
              val manifest = try {
                val ttag = ReflectionUtils.typeTagForType[Any](tpe = argSymbol.info)

              } catch {
                case _ if valueClass.isDefined =>
                  ManifestFactory.classType[Any](ReflectionUtils.classForName(valueClass.get))
              }
              Extraction.decompose(value).extract[Any](formats, mani

     */
  }

  /**
   * Write stage to json
   *
   * @param stage stage instance to write
   * @return write result
   */
  override def write(stage: CustomMonoidAggregator[T]): Try[JValue] = {
    val res = Try {
      serializeFunction("associativeFn", stage.associativeFn)
    }

    res.map(v => {
      ("tto" -> FeatureType.typeName(stage.ttag)) ~
        ("ttov" -> stage.ttov.tpe.typeSymbol.fullName) ~
        ("fn" -> v.value.toString) ~
        ("zero" -> render(Extraction.decompose(stage.zero)))
    })

  }
}