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

/**
 * Define a case class corresponding to our data file (nullable columns must be Option types)
 *
 * @param id       passenger id
 * @param survived 1: survived, 0: did not survive
 * @param pClass   passenger class
 * @param name     passenger name
 * @param sex      passenger sex (male/female)
 * @param age      passenger age (one person has a non-integer age so this must be a double)
 * @param sibSp    number of siblings/spouses traveling with this passenger
 * @param parCh    number of parents/children traveling with this passenger
 * @param ticket   ticket id string
 * @param fare     ticket price
 * @param cabin    cabin id string
 * @param embarked location where passenger embarked
 */
case class IDTextClassification
(
  id: Option[Int],
//  eng_sentences: Option[String],
//  eng_paragraphs: Option[String],
  news_data: Option[String],
  tox_data: Option[String],
  movie_data: Option[String],
  movie_plot: Option[String],
  reddit_science: Option[String],
  fake_id_prefix: Option[String],
  fake_id_sfdc: Option[String],
  fake_uuid: Option[String],
  fake_number_id: Option[String],
  faker_sentence: Option[String],
  faker_paragraph: Option[String],
  faker_ipv4: Option[String],
  faker_ipv6: Option[String],
  faker_id_mix: Option[String]
)

object OpTitanicSimple {

  /**
   * Run this from the command line with
   * ./gradlew sparkSubmit -Dmain=com.salesforce.op.hw.OpTitanicSimple -Dargs=/full/path/to/csv/file
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
    /////////////////////////////////////////////////////////////////////////////////

    // Define features using the OP types based on the data
//    val eng_sentences = FeatureBuilder.Text[IDTextClassification].extract(_.eng_sentences.toText).asPredictor
//    val eng_paragraphs = FeatureBuilder.Text[IDTextClassification].extract(_.eng_paragraphs.toText).asPredictor
    val news_data = FeatureBuilder.Text[IDTextClassification].extract(_.news_data.toText).asPredictor
    val tox_data = FeatureBuilder.Text[IDTextClassification].extract(_.tox_data.toText).asPredictor
    val movie_data = FeatureBuilder.Text[IDTextClassification].extract(_.movie_data.toText).asPredictor
    val movie_plot = FeatureBuilder.Text[IDTextClassification].extract(_.movie_plot.toText).asPredictor
    val reddit_science = FeatureBuilder.Text[IDTextClassification].extract(_.reddit_science.toText).asPredictor
    val fake_id_prefix = FeatureBuilder.Text[IDTextClassification].extract(_.fake_id_prefix.toText).asPredictor
    val fake_id_sfdc = FeatureBuilder.Text[IDTextClassification].extract(_.fake_id_sfdc.toText).asPredictor
    val fake_uuid = FeatureBuilder.Text[IDTextClassification].extract(_.fake_uuid.toText).asPredictor
    val fake_number_id = FeatureBuilder.Text[IDTextClassification].extract(_.fake_number_id.toText).asPredictor
    val faker_sentence = FeatureBuilder.Text[IDTextClassification].extract(_.faker_sentence.toText).asPredictor
    val faker_paragraph = FeatureBuilder.Text[IDTextClassification].extract(_.faker_paragraph.toText).asPredictor
    val faker_ipv4 = FeatureBuilder.Text[IDTextClassification].extract(_.faker_ipv4.toText).asPredictor
    val faker_ipv6 = FeatureBuilder.Text[IDTextClassification].extract(_.faker_ipv6.toText).asPredictor
    val faker_id_mix = FeatureBuilder.Text[IDTextClassification].extract(_.faker_id_mix.toText).asPredictor

    val IDFeatures = Seq(
      // eng_sentences, eng_paragraphs,
      news_data, tox_data,
      movie_data, movie_plot, reddit_science, fake_id_prefix, fake_id_sfdc,
      fake_uuid, fake_number_id, faker_sentence, faker_paragraph,
      faker_ipv4, faker_ipv6, faker_id_mix
    ).transmogrify()

    def thresHoldRFF(mTK: Int): Seq[String] = {
      val dataReader = DataReaders.Simple.csvCase[IDTextClassification](
        path = Option(csvFilePath),
        key = _.id.toString)
      val workflow = new OpWorkflow()
        .withRawFeatureFilter(Some(dataReader), None, minUniqueTokenLen = mTK)
        .setResultFeatures(IDFeatures)
        .setReader(dataReader)

      // Fit the workflow to the data
      val model = workflow.train()
      val rFFresult = workflow.getRawFeatureFilterResults()
      println(rFFresult.rawFeatureDistributions)
      println(rFFresult.rawFeatureFilterMetrics)

      // Extract information (i.e. feature importance) via model insights
      val modelInsights = model.modelInsights(IDFeatures)
      val exclusionReasons = modelInsights.features.flatMap( feature => feature.exclusionReasons)
      exclusionReasons.map(_.name)
    }
    // Stop Spark gracefully
    println(thresHoldRFF(10))
//    println(thresHoldRFF(500))
//    println(thresHoldRFF(1000))
    spark.stop()
  }
}
