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

package com.salesforce.op.cli.gen

import java.io.File

import com.salesforce.op.cli.SchemaSource

import scala.io.Source
import AvroField._

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
  schemaFullName: String,
  theReader: String
)

object ProblemSchema {

  /**
   * Construct a [[ProblemSchema]] from a file containing an avro schema, the name of the response field, and the name
   * of the ID field. May prompt the user for
   *
   * @param schemaSource Contains data schema, can come from a file, or whatever
   * @param responseFieldName The name of the response field (e.g. Passenger survived)
   * @param idFieldName The name o the ID field (e.g. PassengerId)
   * @return The Constructed [[ProblemSchema]]
   */
  def from(ops: Ops, schemaSource: SchemaSource, responseFieldName: String, idFieldName: String): ProblemSchema = {

    // construct avro fields from schema
    val responseField = schemaSource.responseField(responseFieldName)
    val problemKind = responseField.problemKind(ops)
    val idField = schemaSource.idField(idFieldName)

    // construct raw features from avro fields
    val orderedFields = responseField +: schemaSource.fields.diff(responseField::idField::Nil)

    val orf = MakeRawFeature(ops)

    val responseFeature :: features = orderedFields.map { field =>
      orf.from(field, schemaSource.name, field == responseField)
    }

    ProblemSchema(
      kind = problemKind,
      responseFeature = responseFeature,
      idField = idField,
      features = features,
      schemaName = schemaSource.name,
      schemaFullName = schemaSource.fullName,
      theReader = schemaSource.theReader
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
  def scalaCode: String

  /**
   * Gets the java method that Avro generates for this field. e.g. `getPassengerId` for field with name "passengerId"
   * Note that variable names with underscores are converted to CamelCase
   * by avro; so we should do the same
   *
   * @return The java getter as a string
   */
  def avroGetter: String = {
    val pieces = avroField.name.split("_").map(_.capitalize)
    s"get${pieces.mkString("")}"
  }

  /**
   * Gets a name corresponding to the scala `val` that will be generated for this feature builder, e.g.
   * `val myField = FeatureBuilder...`
   *
   * @return The scala val name
   */
  def scalaName: String = avroField.name.toLowerCase

}

trait TemplateBase {
  val templateName: String

  def varname(kind: String): String = s"codeGeneration_${kind}_codeGeneration"

  lazy val sourceCode: String = {
    val ins = getClass.getResourceAsStream(s"/templates/${templateName}Template.scala")
    if (ins == null) {
      throw new UnsupportedOperationException(
        s"Template file ${templateName}Template.scala is missing in resources. " +
          "Probably you will need to rebuild the whole project."
      )
    }
    val allLines = Source.fromInputStream(ins).getLines
    val myLines = allLines
      .dropWhile(s => !(s contains "BEGIN"))
      .drop(1)
      .takeWhile(s => !(s contains "END"))
    myLines.mkString("", "\n", "\n")
  }
  def scalaCode: String = sourceCode
}

case class FeatureListTemplate(whats: Seq[String]) extends TemplateBase {
  override val templateName: String = "FeatureVector"
  val ListName: String = varname("list")
  val ListOfFeatures: String = whats.mkString("Seq(", ",", ")")
  override def scalaCode: String = sourceCode.replaceAll(ListName, ListOfFeatures)
}

case class ProblemTemplate(kind: ProblemKind) extends TemplateBase {
  override val templateName: String = kind.toString
}

abstract class FeatureTemplate(what: String)
  extends OPRawFeature with TemplateBase {
  override val templateName: String = what.capitalize + "Feature"
  protected val SchemaName = "SampleObject"

  val GetterName: String = varname(s"${what}Field")
  override def scalaCode: String =
    sourceCode.replaceAll(SchemaName, schemaName).replaceAll(GetterName, avroGetter) + s".$featureKindString"
}

case class MakeRawFeature(ops: Ops) {

  case class Categorical(avroField: AvroField, schemaName: String, isResponse: Boolean)
    extends FeatureTemplate("categorical")

  case class Text(avroField: AvroField, schemaName: String, isResponse: Boolean)
    extends FeatureTemplate("text")

  case class Real(avroField: AvroField, schemaName: String, isResponse: Boolean)
    extends FeatureTemplate("real")

  case class Binary(avroField: AvroField, schemaName: String, isResponse: Boolean)
    extends FeatureTemplate("binary")

  case class Integral(avroField: AvroField, schemaName: String, isResponse: Boolean)
    extends FeatureTemplate("integral")

  private val featureTypes = List("categorical", "text", "real", "binary", "integral")

  /**
   * Ask the user for the type of a feature representing an avro field.
   *
   * @param field The field to get the type for
   * @param available What available options there are
   * @return What type the user selected
   */
  private def askFeatureType(
    field: AvroField,
    available: List[String] = featureTypes,
    errMsg: String): String = {
    val options = Map(
      "categorical" -> List("categorical", "cat"),
      "text" -> List("text", "string"),
      "real" -> List("real", "continuous"),
      "binary" -> List("binary", "boolean"),
      "integral" -> List("integral", "int")
    ).filter {
      case (featureType, _) => available.contains(featureType)
    }
    ops.ask(s"'${field.name}' - what kind of feature is this?", options, errMsg)
  }

  /**
   * Ask the user whether an integer feature is integral or categorical.
   */
  private def fromInt(field: AvroField, schemaName: String, isResponse: Boolean): OPRawFeature = {
    askFeatureType(
      field,
      List("categorical", "integral"),
      s"Failed to determine the feature kind for $field in $schemaName") match {
      case "categorical" => Categorical(field, schemaName, isResponse)
      case "integral" => Integral(field, schemaName, isResponse)
    }
  }

  /**
   * Ask the user whether a string feature is text or categorical.
   */
  private def fromString(field: AvroField, schemaName: String, isResponse: Boolean): OPRawFeature = {
    askFeatureType(
      field,
      List("categorical", "text"),
      s"Failed to determine the feature kind for $field in $schemaName") match {
      case "categorical" => Categorical(field, schemaName, isResponse)
      case "text" => Text(field, schemaName, isResponse)
    }  }

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
      case AEnum(_, _) => Categorical(field, schemaName, isResponse)
      case AInt(_, _) => fromInt(field, schemaName, isResponse)
      case ABoolean(_, _) => Binary(field, schemaName, isResponse)
      case ALong(_, _) => fromInt(field, schemaName, isResponse)
      case AFloat(_, _) => Real(field, schemaName, isResponse)
      case ADouble(_, _) => Real(field, schemaName, isResponse)
      case AString(_, _) => fromString(field, schemaName, isResponse)
    }
  }

}
