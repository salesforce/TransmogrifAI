/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
sealed trait AvroField {

  /**
   * Whether this schema accepts null values.
   *
   * @return Null value support as a boolean
   */
  def nullable: Boolean

  /**
   * The name of this field
   *
   * @return Field name as string
   */
  def name: String

  /**
   * Figures out what kind of problem is it
   * @param ops environment
   * @return its problem kind
   */
  def problemKind(ops: Ops): ProblemKind
}

/**
 * Operations with AvroFields
 */
object AvroField {

  case class AInt(nullable: Boolean, name: String) extends AvroField {
    def problemKind(ops: Ops): ProblemKind = askKind(ops, this)
  }
  case class ABoolean(nullable: Boolean, name: String) extends AvroField {
    def problemKind(ops: Ops): ProblemKind = BinaryClassification
  }
  case class ALong(nullable: Boolean, name: String) extends AvroField {
    def problemKind(ops: Ops): ProblemKind = Regression
  }
  case class AFloat(nullable: Boolean, name: String) extends AvroField {
    def problemKind(ops: Ops): ProblemKind = Regression
  }
  case class ADouble(nullable: Boolean, name: String) extends AvroField {
    def problemKind(ops: Ops): ProblemKind = Regression
  }
  case class AString(nullable: Boolean, name: String) extends AvroField {
    def problemKind(ops: Ops): ProblemKind =
      askKind(ops, this, List(BinaryClassification, MultiClassification))
  }
  case class AEnum(nullable: Boolean, name: String) extends AvroField {
    def problemKind(ops: Ops): ProblemKind = MultiClassification
  }

  private val AvroTypes: Map[Schema.Type, (Boolean, String) => AvroField] =
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
   * @param nullable Whether the resulting field should be nullable
   * @param name The name of the resulting field
   * @return A `Some[AvroField]` if an avro field could be built.
   */
  private def fromPrimitive(schemaType: Schema.Type, nullable: Boolean, name: String): Option[AvroField] =
    AvroTypes.get(schemaType) map (_(nullable, name))

  private def fromUnion(schema: Schema, name: String): Option[AvroField] = {
    schema.getType match {
      case Schema.Type.UNION =>
        schema.getTypes.asScala.toList.map(_.getType) match {
          case actualType::Schema.Type.NULL::_ => fromPrimitive(actualType, nullable = true, name)
          case Schema.Type.NULL::actualType::_ => fromPrimitive(actualType, nullable = true, name)
          case _ => None
        }
      case _ => None
    }
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
      s => fromPrimitive(s.getType, nullable = false, name)
    }

    def maybeUnion = maybeSchema flatMap (s => fromUnion(s, name))

    maybePrimitive orElse maybeUnion match {
      case Some(fieldSchema) => fieldSchema
      case whatever =>
        throw new IllegalArgumentException(
        s"Type of fieldschema ${field.schema} of field ${field.name} is unsupported. " +
        "Must be a primitive or nullable primitive")
    }
  }
}
