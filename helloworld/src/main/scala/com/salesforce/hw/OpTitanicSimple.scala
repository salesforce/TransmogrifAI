package com.salesforce.hw

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.OpLogisticRegression
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
case class Passenger
(
  id: Int,
  survived: Int,
  pClass: Option[Int],
  name: Option[String],
  sex: Option[String],
  age: Option[Double],
  sibSp: Option[Int],
  parCh: Option[Int],
  ticket: Option[String],
  fare: Option[Double],
  cabin: Option[String],
  embarked: Option[String]
)

/**
 * A sinmplified Optimus Prime example classification app using the Titanic dataset
 */
object OpTitanicSimple {

  /**
   * Run this from the command line with
   * ./gradlew sparkSubmit -Dmain=com.salesforce.hw.OpTitanicSimple -Dargs=/full/path/to/csv/file
   */
  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      println("You need to pass in the CSV file path as an argument")
      sys.exit(1)
    }
    val csvFilePath = args(0)
    println(s"Using user-supplied CSV file path: $csvFilePath")

    // Set up a SparkSession as normal
    val conf = new SparkConf().setAppName(this.getClass.getSimpleName.stripSuffix("$"))
    implicit val spark = SparkSession.builder.config(conf).getOrCreate()

    ////////////////////////////////////////////////////////////////////////////////
    // RAW FEATURE DEFINITIONS
    /////////////////////////////////////////////////////////////////////////////////

    // Define features using the OP types based on the data
    val survived = FeatureBuilder.RealNN[Passenger]
      .extract(_.survived.toRealNN).asResponse

    val pClass = FeatureBuilder.MultiPickList[Passenger]
      .extract(d => d.pClass.map(_.toString).toSet[String].toMultiPickList).asPredictor

    val name = FeatureBuilder.Text[Passenger]
      .extract(d => d.name.toText).asPredictor

    val sex = FeatureBuilder.MultiPickList[Passenger]
      .extract(d => d.sex.map(_.toString).toSet[String].toMultiPickList).asPredictor

    val age = FeatureBuilder.RealNN[Passenger]
      .extract(d => d.age.toRealNN).asPredictor

    val sibSp = FeatureBuilder.Integral[Passenger]
      .extract(d => d.sibSp.toIntegral).asPredictor

    val parCh = FeatureBuilder.Integral[Passenger]
      .extract(d => d.parCh.toIntegral).asPredictor

    val ticket = FeatureBuilder.MultiPickList[Passenger]
      .extract(d => d.ticket.map(_.toString).toSet[String].toMultiPickList).asPredictor

    val fare = FeatureBuilder.Real[Passenger]
      .extract(d => d.fare.toReal).asPredictor

    val cabin = FeatureBuilder.MultiPickList[Passenger]
      .extract(d => d.cabin.map(_.toString).toSet[String].toMultiPickList).asPredictor

    val embarked = FeatureBuilder.MultiPickList[Passenger]
      .extract(d => d.embarked.map(_.toString).toSet[String].toMultiPickList).asPredictor

    ////////////////////////////////////////////////////////////////////////////////
    // TRANSFORMED FEATURES
    /////////////////////////////////////////////////////////////////////////////////

    // Do some basic feature engineering using knowledge of the underlying dataset
    val familySize = sibSp + parCh + 1
    val estimatedCostOfTickets = familySize * fare
    val pivotedSex = sex.pivot()
    val normedAge = age.zNormalize()
    val ageGroup = age.map[MultiPickList](_.value.map(v => if (v > 18) "adult" else "child")
      .toSet[String].toMultiPickList)

    // Define a feature of type vector containing all the predictors you'd like to use
    val passengerFeatures = Seq(
      pClass, name, sex, age, sibSp, parCh, ticket,
      cabin, embarked, familySize, estimatedCostOfTickets,
      pivotedSex, ageGroup).transmogrify()

    // Optionally check the features with a sanity checker
    val sanityCheck = false
    val finalFeatures = if (sanityCheck) survived.sanityCheck(passengerFeatures) else passengerFeatures

    // Define the model we want to use (here a simple logistic regression) and get the resulting output
    val (prediction, rawPrediction, prob) =
      new OpLogisticRegression()
        .setInput(survived, finalFeatures).getOutput

    val evaluator = Evaluators.BinaryClassification()
      .setLabelCol(survived)
      .setRawPredictionCol(rawPrediction)
      .setPredictionCol(prediction)

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW
    /////////////////////////////////////////////////////////////////////////////////

    import spark.implicits._ // Needed for Encoders for the Passenger case class
    // Define a way to read data into our Passenger class from our CSV file
    val trainDataReader = DataReaders.Simple.csvCase[Passenger](
      path = Option(csvFilePath),
      key = _.id.toString
    )

    // Define a new workflow and attach our data reader
    val workflow =
      new OpWorkflow()
        .setResultFeatures(survived, rawPrediction, prob, prediction)
        .setReader(trainDataReader)

    // Fit the workflow to the data
    val fittedWorkflow = workflow.train()
    println(s"Summary: ${fittedWorkflow.summary()}")

    // Manifest the result features of the workflow
    println("Scoring the model")
    val (dataframe, metrics) = fittedWorkflow.scoreAndEvaluate(evaluator = evaluator)

    println("Transformed dataframe columns:")
    dataframe.columns.foreach(println)
    println("Metrics:")
    println(metrics)
  }
}
