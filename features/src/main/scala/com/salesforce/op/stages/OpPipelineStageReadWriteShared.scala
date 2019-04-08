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

import com.salesforce.op.features.FeatureDistributionType
import com.salesforce.op.stages.impl.feature.{HashAlgorithm, HashSpaceStrategy, ScalingType, TimePeriod}
import com.salesforce.op.utils.json.{EnumEntrySerializer, SpecialDoubleSerializer}
import com.salesforce.op.utils.reflection.ReflectionUtils
import enumeratum._
import org.json4s.JsonAST.{JInt, JValue}
import org.json4s.ext.JodaTimeSerializers
import org.json4s.{DefaultFormats, Extraction, Formats}

import scala.reflect._
import scala.reflect.runtime.universe._

object OpPipelineStageReadWriteShared {

  /**
   * Stage json field names
   */
  sealed abstract class FieldNames(override val entryName: String) extends EnumEntry

  /**
   * Stage json field names
   */
  object FieldNames extends Enum[FieldNames] {
    val values = findValues

    case object IsModel extends FieldNames("isModel")

    case object CtorArgs extends FieldNames("ctorArgs")

    case object Uid extends FieldNames("uid")

    case object Class extends FieldNames("class")

    case object ParamMap extends FieldNames("paramMap")

    case object LambdaClassName extends FieldNames("lambdaClassName")

    case object LambdaTypeI1 extends FieldNames("lambdaTypeI1")

    case object LambdaTypeI2 extends FieldNames("lambdaTypeI2")

    case object LambdaTypeI3 extends FieldNames("lambdaTypeI3")

    case object LambdaTypeI4 extends FieldNames("lambdaTypeI4")

    case object LambdaTypeO extends FieldNames("lambdaTypeO")

    case object LambdaTypeOV extends FieldNames("lambdaTypeOV")

    case object LambdaClassArgs extends FieldNames("lambdaClassArgs")

    case object OperationName extends FieldNames("operationName")

  }

  /**
   * Any Value Types
   */
  sealed abstract class AnyValueTypes extends EnumEntry

  /**
   * Any Value Types
   */
  object AnyValueTypes extends Enum[AnyValueTypes] {
    val values = findValues

    case object TypeTag extends AnyValueTypes

    case object Class extends AnyValueTypes

    case object SparkWrappedStage extends AnyValueTypes

    case object Value extends AnyValueTypes

  }

  /**
   * A container for Any Value
   */
  case class AnyValue(`type`: AnyValueTypes, value: Any)

  implicit val formats: Formats =
    DefaultFormats ++
      JodaTimeSerializers.all +
      EnumEntrySerializer.json4s[AnyValueTypes](AnyValueTypes) +
      EnumEntrySerializer.json4s[HashAlgorithm](HashAlgorithm) +
      EnumEntrySerializer.json4s[HashSpaceStrategy](HashSpaceStrategy) +
      EnumEntrySerializer.json4s[ScalingType](ScalingType) +
      EnumEntrySerializer.json4s[TimePeriod](TimePeriod) +
      EnumEntrySerializer.json4s[FeatureDistributionType](FeatureDistributionType) +
      new SpecialDoubleSerializer


  /**
   * convert known types to json
   *
   * @param v Any
   * @return Array ("type","value")
   */
  def valToJson(v: Any): Array[Any] = {
    v match {
      case x: Int => Array("i", x)
      case x: BigDecimal => Array("bd", x)
      case x: BigInt => Array("bi", x)
      case x: Double => Array("d", x)
      case x: Long => Array("l", x)
      case x: String => Array("s", x)
      case x: Boolean => Array("b", x)
      case x: Map[String, _] => Array("m", x.map(t => (t._1, valToJson(t._2))))
      case _ => Array("j", v)
    }
  }


  def jsonToVal(v: Array[JValue], t: Option[TypeTag[Any]] = None): AnyRef = {
    v.head.extract[String] match {
      case "i" => Int.box(v.last.extract[Int])
      case "bd" => v.last.extract[BigDecimal]
      case "bi" => v.last.extract[BigInt]
      case "d" => Double.box(v.last.extract[Double])
      case "s" => v.last.extract[String]
      case "b" => Boolean.box(v.last.extract[Boolean])
      case "l" => Long.box(v.last.extract[Long])
      case "m" => v.last.extract[Map[String, Array[JValue]]].mapValues(x => jsonToVal(x, None))
      case "j" if t.isDefined =>
        Extraction.decompose(v.last).extract[Any](
          formats, ReflectionUtils.manifestForTypeTag[Any](t.get)
        ).asInstanceOf[AnyRef]
      case x => throw new Exception(s"Unsupported type: ${x}")
    }
  }

  /**
   * Load constructor args from known types
   *
   * @param metadataJson
   * @return Array[AnyRef]
   */
  def loadCtorArgs(metadataJson: JValue): Array[AnyRef] = {
    metadataJson.extract[Array[JValue]].map {
      m => jsonToVal(m.extract[Array[JValue]])
    }
  }
}
