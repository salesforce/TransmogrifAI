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

package com.salesforce.op.cli.gen.templates

import java.io.File

import com.salesforce.op.cli.GeneratorConfig
import com.salesforce.op.cli.gen.FileTemplate.Substitutions
import com.salesforce.op.cli.gen._

import scala.util.Random

case class SimpleProject(ops: Ops) extends ProjectTemplate {
  def conf: GeneratorConfig = ops.config
  lazy val schema: ProblemSchema = ops.schema

  def name: String = "simple"

  def welcomeMessage: String = "To get started, read the README.md file in the directory you just created"

  override def replace(f: ProjectFile): ProjectFile = f.path match {
    case "src/main/avro/Passenger.avsc" => f.replaceFrom(conf.schemaFile)
    case "src/main/resources/PassengerData.csv" => f.replaceFrom(conf.inputFile)
    case "project.gitignore" => f.copy(path = ".gitignore")
    case "gradlew" => f.copy(perms = FilePermissions.exec)
    case path if path.endsWith(".notajar") => f.withExtension("jar")
    case _ => f
  }

  override def shouldCopy(path: String): Boolean = path.endsWith(".notajar")

  def props: Substitutions = {
    val random_seed = new Random().nextInt().toString

    val features = (schema.responseFeature +: schema.features).map { feature =>
      s"  val ${feature.scalaName} = ${feature.buildString}"
    }.mkString("{\n\n", "\n\n", "\n\n}")

    val problemKind = schema.kind match {
      case ProblemKind.Regression => "RegressionModelSelector()"
      case ProblemKind.MultiClassification => "MultiClassificationModelSelector()"
      case ProblemKind.BinaryClassification => "BinaryClassificationModelSelector()"
    }

    val featureVectorize = schema.features.map(_.scalaName).mkString(
      "Seq(", ", ", ").vectorize()"
    )

    val keyFn = s"_.get${schema.idField.name.capitalize}.toString"

    val mainClass = s"'com.salesforce.app.${conf.projName}'"

    val modelLocation = new File(conf.projectDirectory, "build/spark/model")
    val scoreLocation = new File(conf.projectDirectory, "build/spark/scores")
    val evalLocation = new File(conf.projectDirectory, "build/spark/eval")
    val metricsLocation = new File(conf.projectDirectory, "build/spark/metrics")
    val readLocation = new File(conf.projectDirectory, "src/main/resources/" + conf.inputFile.getName)

    val readmeOptions = Map(
      "README_MAIN_CLASS" -> s"-Dmain=com.salesforce.app.${conf.projName}",
      "README_MODEL_LOCATION" -> modelLocation.getAbsolutePath,
      "README_READ_LOCATION" -> s"${schema.schemaName}=${readLocation.getAbsolutePath}",
      "README_SCORE_LOCATION" -> scoreLocation.getAbsolutePath,
      "README_EVAL_LOCATION" -> evalLocation.getAbsolutePath,
      "README_METRICS_LOCATION" -> metricsLocation.getAbsolutePath
    )

    readmeOptions ++ Map(
      "RANDOM_SEED" -> random_seed,
      "APP_NAME_LOWER" -> s"'${conf.projName.toLowerCase}'",
      "SCHEMA_IMPORT" -> schema.schemaFullName,
      "SCHEMA_NAME" -> schema.schemaName,
      "KEY_FN" -> keyFn,
      "FEATURES" -> features,
      "MAIN_CLASS" -> mainClass,
      "PROBLEM_KIND" -> problemKind,
      "FEATURE_VECTORIZE" -> featureVectorize,
      "RESPONSE_FEATURE" -> s"Seq(${schema.responseFeature.scalaName}).vectorize()"
    )
  }

}
