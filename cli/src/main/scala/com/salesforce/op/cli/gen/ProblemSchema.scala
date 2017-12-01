/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import java.io.File

import com.salesforce.op.cli.gen.AvroField.ABoolean
import com.salesforce.op.cli.gen.ProblemKind.{BinaryClassification, MultiClassification, Regression, values}
import org.apache.avro.Schema

import collection.JavaConverters._

/**
 * Represents a machine learning problem. The CLI infers what the problem is by looking at schema and data.
 *
 * @param kind What kind of problem is it? Regression, binary classification, multiclassification? A [[ProblemKind]]
 * @param responseFeature A description for the response feature as an [[OPRawFeature]]
 * @param idField The avro field representing object IDs
 * @param features A list of feature descriptions as an [[OPRawFeature]], excluding the id field
 * @param schemaName The class name of the schema
 * @param schemaFullName The full name of the schema, including the package (used for generating import statements)
 */
case class ProblemSchema
(
  kind: ProblemKind,
  responseFeature: OPRawFeature,
  idField: AvroField,
  features: List[OPRawFeature],
  schemaName: String,
  schemaFullName: String
)

object ProblemSchema {

  /**
   * Get _exactly one_ [[AvroField]] by name from a list of the schema's fields. If exactly one isn't found, exit
   * with an error. Not sensitive to case.
   *
   * @param l A list of [[AvroField]]s from the schema's fields
   * @param fieldName The name of the field we're searching for
   * @param fieldType The type of the field we're trying to find exactly one of, typically "response" or "id"
   * @return The [[AvroField]] representing the field we passed in
   */
  private def exactlyOne(l: List[AvroField], fieldName: String, fieldType: String): AvroField = {
    l.filter { _.name.toLowerCase == fieldName } match {
      case Nil => Ops.oops(s"$fieldType field '$fieldName' not found (ignoring case)")
      case schema :: Nil => schema
      case _ => Ops.oops(s"$fieldType field '$fieldName' is defined more than once in the schema (ignoring case)")
    }
  }

  /**
   * Construct a [[ProblemSchema]] from a file containing an avro schema, the name of the response field, and the name
   * of the ID field. May prompt the user for
   *
   * @param schemaFile The file containing an avro schema file (most likely extension .avsc)
   * @param responseFieldName The name of the response field (e.g. Passenger survived)
   * @param idFieldName The name o the ID field (e.g. PassengerId)
   * @return The Constructed [[ProblemSchema]]
   */
  def from(ops: Ops, schemaFile: File, responseFieldName: String, idFieldName: String): ProblemSchema = {

    val parser = new Schema.Parser
    val schema = parser.parse(schemaFile)
    if (schema.getType != Schema.Type.RECORD) {
      Ops.oops(s"Schema '${schema.getFullName}' must be a record schema")
    }

    // construct avro fields from schema
    val fields = schema.getFields.asScala.toList.map(AvroField.from)
    val responseFieldSchema = exactlyOne(fields, responseFieldName.toLowerCase, "Response")
    val idFieldSchema = exactlyOne(fields, idFieldName.toLowerCase(), "Id")

    val problemKind = ProblemKind.from(ops, responseFieldSchema)

    // construct raw features from avro fields
    val orderedFields = responseFieldSchema +: fields.filterNot { field =>
      field == responseFieldSchema || field == idFieldSchema
    }

    val orf = MakeRawFeature(ops)

    val (responseFeature :: features) = orderedFields.map { field =>
      orf.from(field, schema.getName, field == responseFieldSchema)
    }

    ProblemSchema(
      kind = problemKind,
      responseFeature = responseFeature,
      idField = idFieldSchema,
      features = features,
      schemaName = schema.getName,
      schemaFullName = schema.getFullName
    )
  }

}


/**
 * Represents a raw feature inferred from avro schema. This is primarily used for the
 * `buildString` method, which codegens a FeatureBuilder.
 */
sealed trait OPRawFeature {

  /**
   * The [[AvroField]] this raw feature is built from
   *
   * @return The corresponding [[AvroField]]
   */
  def avroField: AvroField

  /**
   * The name of the schema class that features are being built from (e.g. Passenger)
   *
   * @return The name of the schema class
   */
  def schemaName: String

  /**
   * Whether this feature is a response feature.
   *
   * @return `true` if this is a response feature, false otherwise
   */
  def isResponse: Boolean

  /**
   * Whether to call `asResponse` or `asPredictor` on the FeatureBuilder.
   *
   * @return A string representing the method
   */
  def featureKindString: String = if (isResponse) "asResponse" else "asPredictor"

  /**
   * Return the string representing the code that makes a FeatureBuilder
   *
   * @return Generated code
   */
  def buildString: String

  /**
   * Gets the java method that Avro generates for this field. e.g. `getPassengerId` for field with name "passengerId"
   *
   * @return The java getter as a string
   */
  def avroGetter: String = s"get${avroField.name.capitalize}"

  /**
   * Gets a name corresponding to the scala `val` that will be generated for this feature builder, e.g.
   * `val myField = FeatureBuilder...`
   *
   * @return The scala val name
   */
  def scalaName: String = avroField.name.toLowerCase

}

case class MakeRawFeature(ops: Ops) {

  case class Categorical(avroField: AvroField, schemaName: String, isResponse: Boolean) extends OPRawFeature {
    def buildString: String =
      s"FeatureBuilder.Categorical[$schemaName]" +
        s".extract(o => Option(o.$avroGetter).map(_.toString).toSet[String].toCategorical).$featureKindString"
  }

  case class Text(avroField: AvroField, schemaName: String, isResponse: Boolean) extends OPRawFeature {
    def buildString: String =
      s"FeatureBuilder.Text[$schemaName]" +
        s".extract(o => Option(o.$avroGetter).toText).$featureKindString"
  }

  case class Real(avroField: AvroField, schemaName: String, isResponse: Boolean) extends OPRawFeature {
    def buildString: String =
      s"FeatureBuilder.Real[$schemaName]" +
        s".extract(o => Option(o.$avroGetter).map(_.toDouble).toReal).$featureKindString"
  }

  case class Binary(avroField: AvroField, schemaName: String, isResponse: Boolean) extends OPRawFeature {
    def buildString: String =
      s"FeatureBuilder.Binary[$schemaName]" +
        s".extract(o => Option(o.$avroGetter).map(_.booleanValue).toBinary).$featureKindString"
  }

  case class Integral(avroField: AvroField, schemaName: String, isResponse: Boolean) extends OPRawFeature {
    def buildString: String =
      s"FeatureBuilder.Integral[$schemaName]" +
        s".extract(o => Option(o.$avroGetter).map(_.toLong).toIntegral).$featureKindString"
  }

  private val featureTypes = List("categorical", "text", "real", "binary", "integral")

  /**
   * Ask the user for the type of a feature representing an avro field.
   *
   * @param field The field to get the type for
   * @param available What available options there are
   * @return What type the user selected
   */
  private def askFeatureType(field: AvroField, available: List[String] = featureTypes): String = {
    val options = Map(
      "categorical" -> List("categorical", "cat"),
      "text" -> List("text", "string"),
      "real" -> List("real", "continuous"),
      "binary" -> List("binary", "boolean"),
      "integral" -> List("integral", "int")
    ).filter {
      case (featureType, _) => available.contains(featureType)
    }
    ops.ask(s"'${field.name}' - what kind of feature is this?", options)
  }

  /**
   * Ask the user whether an integer feature is integral or categorical.
   */
  private def fromInt(field: AvroField, schemaName: String, isResponse: Boolean): OPRawFeature = {
    askFeatureType(field, List("categorical", "integral")) match {
      case "categorical" => Categorical(field, schemaName, isResponse)
      case "integral" => Real(field, schemaName, isResponse)
    }
  }

  /**
   * Ask the user whether a string feature is text or categorical.
   */
  private def fromString(field: AvroField, schemaName: String, isResponse: Boolean): OPRawFeature = {
    askFeatureType(field, List("categorical", "text")) match {
      case "categorical" => Categorical(field, schemaName, isResponse)
      case "text" => Text(field, schemaName, isResponse)
    }
  }

  /**
   * Construct an [[OPRawFeature]] from an avro field, the class name of the schema (e.g. Passenger), and whether
   * this feature should be a response feature.
   *
   * @param field The underlying [[AvroField]]
   * @param schemaName The name of the schema
   * @param isResponse Whether the generated feature should be a response feature
   * @return An [[OPRawFeature]]
   */
  def from(field: AvroField, schemaName: String, isResponse: Boolean): OPRawFeature = {
    field match {
      case AvroField.AEnum(_, _) => Categorical(field, schemaName, isResponse)
      case AvroField.AInt(_, _) => fromInt(field, schemaName, isResponse)
      case AvroField.ABoolean(_, _) => Binary(field, schemaName, isResponse)
      case AvroField.ALong(_, _) => fromInt(field, schemaName, isResponse)
      case AvroField.AFloat(_, _) => Real(field, schemaName, isResponse)
      case AvroField.ADouble(_, _) => Real(field, schemaName, isResponse)
      case AvroField.AString(_, _) => fromString(field, schemaName, isResponse)
    }
  }

}
