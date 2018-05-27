/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.cli.gen

import collection.JavaConverters._
import org.apache.avro.Schema
import Schema._
import ProblemKind._
import com.salesforce.op.cli.gen.AvroField.AEnum
import com.salesforce.op.cli.gen.ProblemKind.{BinaryClassification, MultiClassification, Regression}


/**
 * Represents a field in an Avro record. Can be nullable.
 */
sealed trait AvroField { field =>

  def schemaField: Schema.Field

  /**
   * Whether this schema accepts null values.
   *
   * @return Null value support as a boolean
   */
  def isNullable: Boolean

  /**
   * The name of this field
   *
   * @return Field name as string
   */
  def name: String = schemaField.name

  /**
   * Figures out what kind of problem is it
   * @param ops environment
   * @return its problem kind
   */
  def problemKind(ops: Ops): ProblemKind = {
    if (isNullable) {
      Ops.oops(s"Response field '$field' cannot be nullable")
    }
    defaultProblemKind(ops)
  }

  protected def defaultProblemKind(ops: Ops): ProblemKind

  override def equals(something: Any): Boolean = something match {
    case af: AvroField => af.name == name
    case other => false
  }

  override def hashCode(): Int = name.hashCode

  override def toString: String = getClass.getSimpleName + s"($name${if (isNullable) ", nullable"})"
}

/**
 * Operations with AvroFields
 */
object AvroField {

  case class AInt(schemaField: Schema.Field, isNullable: Boolean) extends AvroField {
    def defaultProblemKind(ops: Ops): ProblemKind = askKind(ops, this)
  }
  case class ABoolean(schemaField: Schema.Field, isNullable: Boolean) extends AvroField {
    def defaultProblemKind(ops: Ops): ProblemKind = BinaryClassification
  }
  case class ALong(schemaField: Schema.Field, isNullable: Boolean) extends AvroField {
    def defaultProblemKind(ops: Ops): ProblemKind = Regression
  }
  case class AFloat(schemaField: Schema.Field, isNullable: Boolean) extends AvroField {
    def defaultProblemKind(ops: Ops): ProblemKind = Regression
  }
  case class ADouble(schemaField: Schema.Field, isNullable: Boolean) extends AvroField {
    def defaultProblemKind(ops: Ops): ProblemKind = Regression
  }
  case class AString(schemaField: Schema.Field, isNullable: Boolean) extends AvroField {
    def defaultProblemKind(ops: Ops): ProblemKind =
      askKind(ops, this, List(BinaryClassification, MultiClassification))
  }
  case class AEnum(schemaField: Schema.Field, isNullable: Boolean) extends AvroField {
    def defaultProblemKind(ops: Ops): ProblemKind = MultiClassification
  }

  private val AvroTypes: Map[Schema.Type, (Schema.Field, Boolean) => AvroField] =
    Map(
      Type.INT -> AInt,
      Type.BOOLEAN -> ABoolean,
      Type.LONG -> ALong,
      Type.FLOAT -> AFloat,
      Type.DOUBLE -> ADouble,
      Type.STRING -> AString,
      Type.ENUM -> AEnum
    )

  /**
   * Build an [[AvroField]] from a [[Schema.Type]] (an avro-internal class representing a type in the schema).
   * This is called "fromPrimitive" because it only accepts primitive avro types, not unions or records.
   *
   * @param schemaType The [[Schema.Type]] to build an avro field from
   * @param isNullable Whether the resulting field should be nullable
   * @param schemaField The avro schema field
   * @return A `Some[AvroField]` if an avro field could be built.
   */
  private def fromPrimitive(
    schemaType: Schema.Type,
    schemaField: Schema.Field,
    isNullable: Boolean): Option[AvroField] =
    AvroTypes.get(schemaType) map (_(schemaField, isNullable))

  def typeOfNullable(schema: Schema): Option[Schema.Type] = {
    schema.getType match {
      case Schema.Type.UNION =>
        schema.getTypes.asScala.toList.map(_.getType) match {
          case actualType :: Schema.Type.NULL :: _ => Option(actualType)
          case Schema.Type.NULL :: actualType :: _ => Option(actualType)
          case _ => None
        }
      case _ => None
    }
  }

  private def fromUnion(schemaField: Schema.Field): Option[AvroField] = {
    for {
      actualType <- typeOfNullable(schemaField.schema)
      avroField <- fromPrimitive(actualType, schemaField, isNullable = true)
    } yield avroField
  }

  /**
   * Build an [[AvroField]] from a [[Schema.Type]]. This accepts primitive types, or unions with a primitive type and
   * [[Schema.Type.NULL]].
   *
   * @param field The avro schema representing a field
   * @return The constructed [[AvroField]]
   */
  def from(field: Schema.Field): AvroField = {
    val name = field.name()
    val maybeSchema = Option(field.schema())

    def maybePrimitive = maybeSchema flatMap {
      s => fromPrimitive(s.getType, field, isNullable = false)
    }

    def maybeUnion = maybeSchema flatMap (s => fromUnion(field))

    maybePrimitive orElse maybeUnion match {
      case Some(fieldSchema) => fieldSchema
      case whatever =>
        throw new IllegalArgumentException(
        s"Type of fieldschema ${field.schema} of field ${field.name} is unsupported. " +
        "Must be a primitive or nullable primitive")
    }
  }
}
