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

package com.salesforce.hw

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StringType, StructField, StructType}

import scala.io.Source

/**
 * A simplified TransmogrifAI example to test name identification
 */
object OpNameGenderDetection {

  /**
   * Run this from the command line with
   * ./gradlew sparkSubmit -Dmain=com.salesforce.hw.OpNameGenderDetection -Dargs=/full/path/to/csv/file
   */
  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      println("You need to pass in the CSV file path as an argument")
      sys.exit(1)
    }
    val csvFilePath = args(0)
    println(s"Using user-supplied CSV file path: $csvFilePath")

    // Set up a SparkSession as normal
    implicit val spark = SparkSession.builder.config(new SparkConf()).getOrCreate()
    import spark.implicits._ // Needed for Encoders for the Passenger case class

    ////////////////////////////////////////////////////////////////////////////////
    // RAW FEATURE DEFINITIONS
    ////////////////////////////////////////////////////////////////////////////////

    // Check number of columns in CSV file and make appropriate number of Text feature vectors using schema
    val numCols: Int = {
      val src = Source.fromFile(csvFilePath)
      val firstLine: Option[String] = try {
        src.getLines.find(_ => true)
      } finally {
        src.close()
      }
      firstLine match {
        case Some(line) => line.split(",").length
        case None => 0
      }
    }
    val schema = StructType(List.range(1, numCols).map(i => StructField(s"field_$i", StringType, true)))
    val (response, rawFeatures) = FeatureBuilder.fromSchema(schema, "field0")

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW
    /////////////////////////////////////////////////////////////////////////////////

    // Define a way to read data into our Passenger class from our CSV file
    // TODO: Noooooo I still have to define a case class for the specific file
    val dataReader = DataReaders.Simple.csvCase[Passenger](path = Option(csvFilePath), key = _.id.toString)

    // Define a new workflow and attach our data reader
    val workflow = new OpWorkflow().setResultFeatures(survived, prediction).setReader(dataReader)

    // Add in my name estimators
    // Fit the workflow to the data
    val model = workflow.train()

    // Stop Spark gracefully
    spark.stop()
  }
}
