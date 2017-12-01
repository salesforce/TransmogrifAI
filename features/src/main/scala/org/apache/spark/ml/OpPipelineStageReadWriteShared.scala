/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package org.apache.spark.ml

import com.salesforce.op.utils.json.{EnumEntrySerializer, SpecialDoubleSerializer}
import enumeratum._
import org.json4s.{DefaultFormats, Formats}


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
    case object Value extends AnyValueTypes
  }

  /**
   * A container for Any Value
   */
  case class AnyValue(`type`: AnyValueTypes, value: Any)

  implicit val formats: Formats =
    DefaultFormats + EnumEntrySerializer[AnyValueTypes](AnyValueTypes) + new SpecialDoubleSerializer

}
