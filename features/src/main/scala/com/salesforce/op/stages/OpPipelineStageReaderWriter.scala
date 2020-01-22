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
import com.salesforce.op.stages.impl.feature._
import com.salesforce.op.utils.json.{EnumEntrySerializer, SpecialDoubleSerializer}
import com.salesforce.op.utils.reflection.ReflectionUtils
import enumeratum.{Enum, EnumEntry}
import org.json4s.ext.JodaTimeSerializers
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.json4s.{Extraction, Formats, FullTypeHints, JValue}
import org.slf4j.LoggerFactory

import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

/**
 * Value reader/writer implementation used to (de)serialize stage arguments from/to trained models
 *
 * @tparam T value type to read/write
 */
trait ValueReaderWriter[T <: AnyRef] {

  /**
   * Read value from json
   *
   * @param valueClass value class
   * @param json       json to read argument value from
   * @return read result
   */
  def read(valueClass: Class[T], json: JValue): Try[T]

  /**
   * Write value to json
   *
   * @param value value to write
   * @return write result
   */
  def write(value: T): Try[JValue]

}


object ValueReaderWriter {

  private val log = LoggerFactory.getLogger(ValueReaderWriter.getClass)

  /**
   * Retrieve reader/writer implementation: either the custom one specified with [[ReaderWriter]] annotation
   * or the default one [[DefaultValueReaderWriter]]
   *
   * @param valueClass value class
   * @param valueName  value name
   * @tparam T value type
   * @return reader/writer implementation
   */
  def readerWriterFor[T <: AnyRef : ClassTag](valueClass: Class[T], valueName: String): ValueReaderWriter[T] = {
    readerWriterForOrDefault[T, ValueReaderWriter[T]](valueClass, new DefaultValueReaderWriter[T](valueName))
  }

  /**
   * Retrieve reader/writer implementation: either the custom one specified with [[ReaderWriter]] annotation
   * or the default one
   *
   * @param valueClass stage class
   * @param defaultReaderWriter default reader writer instance maker
   * @tparam T value type
   * @return reader/writer implementation
   */
  private[op] def readerWriterForOrDefault[T <: AnyRef : ClassTag, RW  <: ValueReaderWriter[T]]
  (
    valueClass: Class[T],
    defaultReaderWriter: => RW
  ): RW = {
    if (!valueClass.isAnnotationPresent(classOf[ReaderWriter])) defaultReaderWriter
    else {
      Try {
        val readerWriterClass = valueClass.getAnnotation[ReaderWriter](classOf[ReaderWriter]).value()
        ReflectionUtils.newInstance[RW](readerWriterClass.getName)
      } match {
        case Success(readerWriter) =>
          if (log.isDebugEnabled) {
            log.debug(s"Using reader/writer of type '${readerWriter.getClass.getName}'"
              + s"to (de)serialize value of type '${valueClass.getName}'")
          }
          readerWriter
        case Failure(e) => throw new RuntimeException(
          s"Failed to create reader/writer instance for value class ${valueClass.getName}", e)
      }
    }
  }

}


/**
 * Stage reader/writer implementation used to (de)serialize stages from/to trained models
 *
 * @tparam StageType stage type to read/write
 */
trait OpPipelineStageReaderWriter[StageType <: OpPipelineStageBase]
  extends ValueReaderWriter[StageType] with OpPipelineStageReadWriteFormats


object OpPipelineStageReaderWriter extends OpPipelineStageReadWriteFormats {

  /**
   * Retrieve reader/writer implementation: either the custom one specified with [[ReaderWriter]] annotation
   * or the default one [[DefaultOpPipelineStageReaderWriter]]
   *
   * @param stageClass stage class
   * @tparam StageType stage type
   * @return reader/writer implementation
   */
  def readerWriterFor[StageType <: OpPipelineStageBase : ClassTag]
  (
    stageClass: Class[StageType]
  ): OpPipelineStageReaderWriter[StageType] = {
    ValueReaderWriter.readerWriterForOrDefault[StageType, OpPipelineStageReaderWriter[StageType]](
      stageClass, defaultReaderWriter = new DefaultOpPipelineStageReaderWriter[StageType]()
    )
  }

  /**
   * Serialize value to json
   */
  def jsonSerialize(v: Any): JValue = render(Extraction.decompose(v)(formats))(formats)

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
    case object DefaultParamMap extends FieldNames("defaultParamMap")
    case object Timestamp extends FieldNames("timestamp")
    case object SparkVersion extends FieldNames("sparkVersion")
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
    case object SparkWrappedStage extends AnyValueTypes
    case object ClassInstance extends AnyValueTypes
    case object Value extends AnyValueTypes
  }

  /**
   * A container for Any Value
   */
  case class AnyValue(`type`: AnyValueTypes, value: Any, valueClass: Option[String])

}


trait OpPipelineStageReadWriteFormats {

  import OpPipelineStageReaderWriter._

  val typeHints = FullTypeHints(List(
    classOf[EmptyScalerArgs], classOf[LinearScalerArgs]
  ))

  implicit val formats: Formats =
    Serialization.formats(typeHints) ++
      JodaTimeSerializers.all +
      EnumEntrySerializer.json4s[AnyValueTypes](AnyValueTypes) +
      EnumEntrySerializer.json4s[HashAlgorithm](HashAlgorithm) +
      EnumEntrySerializer.json4s[HashSpaceStrategy](HashSpaceStrategy) +
      EnumEntrySerializer.json4s[TextVectorizationMethod](TextVectorizationMethod) +
      EnumEntrySerializer.json4s[ScalingType](ScalingType) +
      EnumEntrySerializer.json4s[TimePeriod](TimePeriod) +
      EnumEntrySerializer.json4s[FeatureDistributionType](FeatureDistributionType) +
      EnumEntrySerializer.json4s[CombinationStrategy](CombinationStrategy) +
      GenderDetectStrategy.json4s +
      new SpecialDoubleSerializer

}


private[op] trait OpPipelineStageSerializationFuns {

  import OpPipelineStageReaderWriter._

  def serializeArgument(argName: String, value: AnyRef): AnyValue = {
    try {
      val valueClass = value.getClass
      // Test that value has no external dependencies and can be constructed without ctor args or is an object
      ReflectionUtils.newInstance[AnyRef](valueClass.getName)
      AnyValue(AnyValueTypes.ClassInstance, valueClass.getName, Option(valueClass.getName))
    } catch {
      case error: Exception => throw new RuntimeException(
        s"Argument '$argName' [${value.getClass.getName}] cannot be serialized. " +
          s"Make sure ${value.getClass.getName} has either no-args ctor or is an object, " +
          "and does not have any external dependencies, e.g. use any out of scope variables.", error)
    }
  }

}
