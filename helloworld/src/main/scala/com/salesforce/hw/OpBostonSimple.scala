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
import com.salesforce.op.stages.impl.regression.RegressionModelSelector
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
  * Define a case class representing the Boston housing data
  *
  * @param rowId   id of the house
  * @param crim    per capita crime rate by town
  * @param zn      proportion of residential land zoned for lots over 25,000 sq.ft.
  * @param indus   proportion of non-retail business acres per town
  * @param chas    Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
  * @param nox     nitric oxides concentration (parts per 10 million)
  * @param rm      average number of rooms per dwelling
  * @param age     proportion of owner-occupied units built prior to 1940
  * @param dis     weighted distances to five Boston employment centres
  * @param rad     index of accessibility to radial highways
  * @param tax     full-value property-tax rate per $10,000
  * @param ptratio pupil-teacher ratio by town
  * @param b       1000(Bk - 0.63)**2 where Bk is the proportion of blacks by town
  * @param lstat   % lower status of the population
  * @param medv    median value of owner-occupied homes in $1000's
  */
case class BostonHouse
(
  rowId: Int,
  crim: Double,
  zn: Double,
  indus: Double,
  chas: String,
  nox: Double,
  rm: Double,
  age: Double,
  dis: Double,
  rad: Int,
  tax: Double,
  ptratio: Double,
  b: Double,
  lstat: Double,
  medv: Double
)

/**
 * A simplified TransmogrifAI example classification app using the Boston dataset
 */
object OpBostonSimple {

  /**
   * Run this from the command line with
   * ./gradlew sparkSubmit -Dmain=com.salesforce.hw.OpBostonSimple -Dargs=/full/path/to/csv/file
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
    import spark.implicits._ // Needed for Encoders for the BostonHouse case class

    ////////////////////////////////////////////////////////////////////////////////
    // RAW FEATURE DEFINITIONS
    /////////////////////////////////////////////////////////////////////////////////

    // Define features using the OP types based on the data
    val rowId = FeatureBuilder.Integral[BostonHouse].extract(_.rowId.toIntegral).asPredictor
    val crim = FeatureBuilder.RealNN[BostonHouse].extract(_.crim.toRealNN).asPredictor
    val zn = FeatureBuilder.RealNN[BostonHouse].extract(_.zn.toRealNN).asPredictor
    val indus = FeatureBuilder.RealNN[BostonHouse].extract(_.indus.toRealNN).asPredictor
    val chas = FeatureBuilder.PickList[BostonHouse].extract(x => Option(x.chas).toPickList).asPredictor
    val nox = FeatureBuilder.RealNN[BostonHouse].extract(_.nox.toRealNN).asPredictor
    val rm = FeatureBuilder.RealNN[BostonHouse].extract(_.rm.toRealNN).asPredictor
    val age = FeatureBuilder.RealNN[BostonHouse].extract(_.age.toRealNN).asPredictor
    val dis = FeatureBuilder.RealNN[BostonHouse].extract(_.dis.toRealNN).asPredictor
    val rad = FeatureBuilder.Integral[BostonHouse].extract(_.rad.toIntegral).asPredictor
    val tax = FeatureBuilder.RealNN[BostonHouse].extract(_.tax.toRealNN).asPredictor
    val ptratio = FeatureBuilder.RealNN[BostonHouse].extract(_.ptratio.toRealNN).asPredictor
    val b = FeatureBuilder.RealNN[BostonHouse].extract(_.b.toRealNN).asPredictor
    val lstat = FeatureBuilder.RealNN[BostonHouse].extract(_.lstat.toRealNN).asPredictor
    val medv = FeatureBuilder.RealNN[BostonHouse].extract(_.medv.toRealNN).asResponse


    // Define a feature of type vector containing all the predictors you'd like to use
    val features = Seq(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat).transmogrify()

    val labels = medv

    val checkedFeatures = labels.sanityCheck(features, removeBadFeatures = true)

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW DEFINITION
    /////////////////////////////////////////////////////////////////////////////////

    // Define the model we want to use (here a simple linear regression) and get the resulting output

    val prediction = RegressionModelSelector
      .withTrainValidationSplit(
        modelTypesToUse = Seq(OpLinearRegression))
      .setInput(labels, checkedFeatures).getOutput()

    val evaluator = Evaluators.Regression().setLabelCol(labels).setPredictionCol(prediction)

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW
    /////////////////////////////////////////////////////////////////////////////////

    val dataReader = DataReaders.Simple.csvCase[BostonHouse](path = Option(csvFilePath), key = _.rowId.toString())

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
