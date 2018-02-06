/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen.templates

import java.io.File

import com.salesforce.op.cli.{AvroSchemaFromFile, GeneratorConfig}
import com.salesforce.op.cli.gen.FileGenerator.Substitutions
import com.salesforce.op.cli.gen._

import scala.util.Random

case class SimpleProject(ops: Ops) extends ProjectGenerator {
  def conf: GeneratorConfig = ops.config
  /**
   * The [[ProblemSchema]] for this command to generate a project. This is a lazy val so we don't prompt the user
   * until it is actually needed.
   */
  lazy val schema: ProblemSchema = ProblemSchema.from(ops, conf.schemaSource, conf.response, conf.idField)

  def name: String = "simple"

  def welcomeMessage: String = "To get started, read the README.md file in the directory you just created"

  lazy val schemaFile: File = conf.schemaSource.schemaFile

  override def replace(f: FileInProject): FileInProject = f.path match {
    case "src/main/avro/Passenger.avsc" => f.replaceFrom(schemaFile)
    case "project.gitignore" => f.copy(path = ".gitignore")
    case "gradlew" => f.copy(perms = FilePermissions.exec)
    case path if path.endsWith(".notajar") => f.withExtension("jar")
    case _ => f
  }

  override def shouldCopy(path: String): Boolean = path.endsWith(".notajar")

  def props: Substitutions = {
    val random_seed = new Random().nextInt().toString

    val features = (schema.responseFeature +: schema.features).map { feature =>
      s"  val ${feature.scalaName} = ${feature.scalaCode}"
    }.mkString("{\n\n", "\n\n", "\n\n}")

    val problemKind = ProblemTemplate(schema.kind).scalaCode

    val featureList = FeatureListTemplate(schema.features.map(_.scalaName)).scalaCode

    val keyFn = s"_.get${schema.idField.name.capitalize}.toString"

    val mainClass = s"'com.salesforce.app.${conf.projName}'"

    val modelLocation = new File(conf.projectDirectory, "build/spark/model")
    val scoreLocation = new File(conf.projectDirectory, "build/spark/scores")
    val evalLocation = new File(conf.projectDirectory, "build/spark/eval")
    val metricsLocation = new File(conf.projectDirectory, "build/spark/metrics")

    val readerChoice = schema.theReader

    val readmeOptions = Map(
      "README_MAIN_CLASS" -> s"-Dmain=com.salesforce.app.${conf.projName}",
      "README_MODEL_LOCATION" -> modelLocation.getAbsolutePath,
      "README_READ_LOCATION" -> s"${schema.schemaName}=${conf.inputFile.getAbsolutePath}",
      "README_SCORE_LOCATION" -> scoreLocation.getAbsolutePath,
      "README_EVAL_LOCATION" -> evalLocation.getAbsolutePath,
      "README_METRICS_LOCATION" -> metricsLocation.getAbsolutePath
    )


    val allProps = readmeOptions ++ Map(
      "RANDOM_SEED" -> random_seed,
      "APP_NAME_LOWER" -> s"'${conf.projName.toLowerCase}'",
      "SCHEMA_IMPORT" -> schema.schemaFullName,
      "SCHEMA_NAME" -> schema.schemaName,
      "KEY_FN" -> keyFn,
      "FEATURES" -> features,
      "MAIN_CLASS" -> mainClass,
      "PROBLEM_KIND" -> problemKind,
      "FEATURE_LIST" -> featureList,
      "RESPONSE_FEATURE" -> FeatureListTemplate(schema.responseFeature.scalaName::Nil).scalaCode,
      "READER_CHOICE" -> readerChoice
    )

    allProps
  }

}
