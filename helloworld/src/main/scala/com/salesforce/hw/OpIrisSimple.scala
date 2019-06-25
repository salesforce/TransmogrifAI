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
import com.salesforce.op.stages.impl.classification.MultiClassificationModelSelector
import com.salesforce.op.stages.impl.classification.MultiClassClassificationModelsToTry._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
  * Define a case class corresponding to our data
  * @param id           flower id
  * @param sepalLength  sepal length in cm
  * @param sepalWidth   sepal width in cm
  * @param petalLength  petal length in cm
  * @param petalWidth   petal width in cm
  * @param irisClass    class of iris (Iris Setosa, Iris Veriscolour, Iris Virginica)
  */
case class Iris
(
  id: Int,
  sepalLength: Double,
  sepalWidth: Double,
  petalLength: Double,
  petalWidth: Double,
  irisClass: String
)

/**
 * A simplified TransmogrifAI example classification app using the Iris dataset
 */
object OpIrisSimple {

  /**
   * Run this from the command line with
   * ./gradlew sparkSubmit -Dmain=com.salesforce.hw.OpIrisSimple -Dargs=/full/path/to/csv/file
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
    import spark.implicits._ // Needed for Encoders for the Iris case class

    ////////////////////////////////////////////////////////////////////////////////
    // RAW FEATURE DEFINITIONS
    /////////////////////////////////////////////////////////////////////////////////

    // Define features using the OP types based on the data
    val sepalLength = FeatureBuilder.Real[Iris].extract(_.sepalLength.toReal).asPredictor
    val sepalWidth = FeatureBuilder.Real[Iris].extract(_.sepalWidth.toReal).asPredictor
    val petalLength = FeatureBuilder.Real[Iris].extract(_.petalLength.toReal).asPredictor
    val petalWidth = FeatureBuilder.Real[Iris].extract(_.petalWidth.toReal).asPredictor
    val irisClass = FeatureBuilder.Text[Iris].extract(_.irisClass.toText).asResponse


    // Define a feature of type vector containing all the predictors you'd like to use
    val features = Seq(sepalLength, sepalWidth, petalLength, petalWidth).transmogrify()

    val labels = irisClass.indexed()

    val checkedFeatures = labels.sanityCheck(features, removeBadFeatures = true)


    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW DEFINITION
    /////////////////////////////////////////////////////////////////////////////////

    // Define the model we want to use (here a simple logistic regression) and get the resulting output

    val prediction = MultiClassificationModelSelector
      .withTrainValidationSplit(
        modelTypesToUse = Seq(OpLogisticRegression))
      .setInput(labels, checkedFeatures).getOutput()

    val evaluator = Evaluators.MultiClassification().setLabelCol(labels).setPredictionCol(prediction)

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW
    /////////////////////////////////////////////////////////////////////////////////

    val dataReader = DataReaders.Simple.csvCase[Iris](path = Option(csvFilePath), key = _.id.toString())

    val workflow = new OpWorkflow().setResultFeatures(prediction, labels).setReader(dataReader)

    val model = workflow.train()

    println(s"Model summary:\n${model.summaryPretty()}")

    // Manifest the result features of the workflow
    println("Scoring the model")
    val (scores, metrics) = model.scoreAndEvaluate(evaluator = evaluator)

    println("Metrics:\n" + metrics)

    // Stop Spark gracefully
    spark.stop()
  }
}
