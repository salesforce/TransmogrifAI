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

package com.salesforce.hw.titanic

import com.salesforce.op._
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry.{OpLogisticRegression, OpRandomForestClassifier}
import com.salesforce.op.stages.impl.classification._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
 * A minimal Titanic Survival example with TransmogrifAI
 */
object OpTitanicMini {

  case class Passenger
  (
    id: Long,
    survived: Double,
    pClass: Option[Long],
    name: Option[String],
    sex: Option[String],
    age: Option[Double],
    sibSp: Option[Long],
    parCh: Option[Long],
    ticket: Option[String],
    fare: Option[Double],
    cabin: Option[String],
    embarked: Option[String]
  )

  def main(args: Array[String]): Unit = {
    implicit val spark = SparkSession.builder.config(new SparkConf()).getOrCreate()
    import spark.implicits._

    // Read Titanic data as a DataFrame
    val pathToData = Option(args(0))
    val passengersData = DataReaders.Simple.csvCase[Passenger](pathToData, key = _.id.toString).readDataset().toDF()

    // Automated feature engineering
    val (survived, features) = FeatureBuilder.fromDataFrame[RealNN](passengersData, response = "survived")
    val featureVector = features.transmogrify()

    // Automated feature selection
    val checkedFeatures = survived.sanityCheck(featureVector, checkSample = 1.0, removeBadFeatures = true)

    // Automated model selection
    val prediction = BinaryClassificationModelSelector
      .withCrossValidation(modelTypesToUse = Seq(OpLogisticRegression, OpRandomForestClassifier))
      .setInput(survived, checkedFeatures).getOutput()

    val model = new OpWorkflow().setInputDataset(passengersData).setResultFeatures(prediction).train()

    println("Model summary:\n" + model.summaryPretty())
  }

}
