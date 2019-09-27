package com.salesforce.op.filters

import com.salesforce.op._
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataReaders
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
 * @author nguyen.tuan
 * @since 214
 */

case class IDTextClassification
(
  eng_sentences: Option[String],
  eng_paragraphs: Option[String],
  news_data: Option[String],
  tox_data: Option[String],
  movie_data: Option[String],
  movie_plot: Option[String],
  reddit_science: Option[String],
  fake_id_prefix: Option[String],
  alpha_numeric: Option[String],
  fake_id_sfdc: Option[String],
  fake_uuid: Option[String],
  fake_number_id: Option[String],
  faker_sentence: Option[String],
  variable_nb_words: Option[String],
  faker_paragraph: Option[String],
  variable_nb_sentences: Option[String],
  faker_ipv4: Option[String],
  faker_ipv6: Option[String]
)

object IDClassification {

  /**
   * Run this from the command line with
   * ./gradlew sparkSubmit -Dmain=com.salesforce.hw.OpTitanicSimple -Dargs=/full/path/to/csv/file
   */
  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      println("You need to pass in the CSV file path as an argument")
      sys.exit(1)
    }
    // TODO: replace this with the path to CSV file of all text fields
    val csvFilePath = args(0)
    println(s"Using user-supplied CSV file path: $csvFilePath")

    // Set up a SparkSession as normal
    implicit val spark = SparkSession.builder.config(new SparkConf()).getOrCreate()
    import spark.implicits._ // Needed for Encoders for the Passenger case class

    ////////////////////////////////////////////////////////////////////////////////
    // RAW FEATURE DEFINITIONS
    /////////////////////////////////////////////////////////////////////////////////

    // Define features using the OP types based on the data
    val eng_sentences = FeatureBuilder.Text[IDTextClassification].extract(_.eng_sentences.toText).asPredictor
    val eng_paragraphs = FeatureBuilder.Text[IDTextClassification].extract(_.eng_paragraphs.toText).asPredictor
    val news_data = FeatureBuilder.Text[IDTextClassification].extract(_.news_data.toText).asPredictor
    val tox_data = FeatureBuilder.Text[IDTextClassification].extract(_.tox_data.toText).asPredictor
    val movie_data = FeatureBuilder.Text[IDTextClassification].extract(_.movie_data.toText).asPredictor
    val movie_plot = FeatureBuilder.Text[IDTextClassification].extract(_.movie_plot.toText).asPredictor
    val reddit_science = FeatureBuilder.Text[IDTextClassification].extract(_.reddit_science.toText).asPredictor
    val fake_id_prefix = FeatureBuilder.Text[IDTextClassification].extract(_.fake_id_prefix.toText).asPredictor
    val alpha_numeric = FeatureBuilder.Text[IDTextClassification].extract(_.alpha_numeric.toText).asPredictor
    val fake_id_sfdc = FeatureBuilder.Text[IDTextClassification].extract(_.fake_id_sfdc.toText).asPredictor
    val fake_uuid = FeatureBuilder.Text[IDTextClassification].extract(_.fake_uuid.toText).asPredictor
    val fake_number_id = FeatureBuilder.Text[IDTextClassification].extract(_.fake_number_id.toText).asPredictor
    val faker_sentence = FeatureBuilder.Text[IDTextClassification].extract(_.faker_sentence.toText).asPredictor
    val variable_nb_words = FeatureBuilder.Text[IDTextClassification].extract(_.variable_nb_words.toText).asPredictor
    val faker_paragraph = FeatureBuilder.Text[IDTextClassification].extract(_.faker_paragraph.toText).asPredictor
    val variable_nb_sentences = FeatureBuilder.Text[IDTextClassification]
      .extract(_.variable_nb_sentences.toText).asPredictor
    val faker_ipv4 = FeatureBuilder.Text[IDTextClassification].extract(_.faker_ipv4.toText).asPredictor
    val faker_ipv6 = FeatureBuilder.Text[IDTextClassification].extract(_.faker_ipv6.toText).asPredictor

    val IDFeatures = Seq(
      eng_sentences, eng_paragraphs, news_data, tox_data,
      movie_data, movie_plot, reddit_science, fake_id_prefix,
      alpha_numeric, fake_id_sfdc, fake_uuid, fake_number_id,
      faker_sentence, variable_nb_words, faker_paragraph,
      variable_nb_sentences, faker_ipv4, faker_ipv6
    ).transmogrify()

    val dataReader = DataReaders.Simple.csvCase[IDTextClassification](path = Option(csvFilePath), key = _.id.toString)
    val workflow = new OpWorkflow()
      .withRawFeatureFilter(Some(dataReader), None)
      .setResultFeatures(IDFeatures)
      .setReader(dataReader)

    // Fit the workflow to the data
    val model = workflow.train()
    println(s"Model summary:\n${model.summaryPretty()}")

    // Extract information (i.e. feature importance) via model insights
    val modelInsights = model.modelInsights(IDFeatures)
    val exclusionReasons = modelInsights.features.flatMap( feature => feature.exclusionReasons)
    val modelFeatures = modelInsights.features.flatMap( feature => feature.derivedFeatures)
    // Stop Spark gracefully
    spark.stop()
  }
}

