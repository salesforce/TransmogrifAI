/* setFileName APP_NAME */
/* replace Passenger SCHEMA_NAME */
package com.salesforce.app

import com.salesforce.app.schema.Passenger /* << SCHEMA_IMPORT */
import com.salesforce.op._
import com.salesforce.op.readers._
import com.salesforce.op.evaluators._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.preparators._
import com.salesforce.op.stages.impl.regression._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}


object /* APP_NAME >> */ Simple extends OpApp with Features {

  ////////////////////////////////////////////////////////////////////////////////
  // READER DEFINITIONS
  /////////////////////////////////////////////////////////////////////////////////

  val trainingReader = new CSVReader[Passenger](
    readPath = None,
    schema = Passenger.getClassSchema.toString,
    key = _.getPassengerId.toString /* << KEY_FN */
  ) {
    override def read(params: OpParams)(implicit spark: SparkSession): Either[RDD[Passenger], Dataset[Passenger]] = {
      split(data = super.read(params), isTest = false)
    }
  }

  val scoringReader = new CSVReader[Passenger](
    readPath = None,
    schema = Passenger.getClassSchema.toString,
    key = _.getPassengerId.toString /* << KEY_FN */
  ) {
    override def read(params: OpParams)(implicit spark: SparkSession): Either[RDD[Passenger], Dataset[Passenger]] = {
      split(data = super.read(params), isTest = true)
    }
  }

  val randomSeed = 42 /* << RANDOM_SEED */

  private def split(
    data: Either[RDD[Passenger], Dataset[Passenger]],
    isTest: Boolean,
    weights: Array[Double] = Array(0.9, 0.1)
  ): Either[RDD[Passenger], Dataset[Passenger]] = data match {
    case Left(rdd) =>
      val Array(train, test) = rdd.randomSplit(weights, randomSeed)
      Left(if (isTest) test else train)
    case Right(ds) =>
      val Array(train, test) = ds.randomSplit(weights, randomSeed)
      Right(if (isTest) test else train)
  }


  ////////////////////////////////////////////////////////////////////////////////
  // WORKFLOW DEFINITION
  /////////////////////////////////////////////////////////////////////////////////

  val featureVector =
    Seq(pClass, name, sex, age, sibSp, parch, ticket, cabin, embarked).transmogrify() /* << FEATURE_VECTORIZE */

  val labelFixed =
    survived.vectorize() /* << RESPONSE_FEATURE */
    .map[RealNN](_.value(0).toRealNN)

  val checkedFeatures = new SanityChecker()
    .setCheckSample(0.10)
    .setInput(labelFixed, featureVector)
    .getOutput()

  val (pred, raw, prob) = BinaryClassificationModelSelector() /* << PROBLEM_KIND */
    .setInput(labelFixed, checkedFeatures)
    .getOutput()

  val workflow = new OpWorkflow().setResultFeatures(pred)

  val evaluator =
    Evaluators.BinaryClassification()
      .setLabelCol(labelFixed).setPredictionCol(pred).setRawPredictionCol(raw)

  def run(runType: OpWorkflowRunType, opParams: OpParams)(implicit spark: SparkSession): Unit = {
    new OpWorkflowRunner(
      workflow = workflow,
      trainingReader = trainingReader,
      scoringReader = scoringReader,
      evaluator = evaluator,
      scoringEvaluator = Some(evaluator),
      featureToComputeUpTo = featureVector
    ).run(runType, opParams)
  }

}
