package com.salesforce.hw.titanic

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.readers.CSVReader
import com.salesforce.op.stages.impl.classification.ClassificationModelsToTry._
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.preparators.SanityChecker
import com.salesforce.op.stages.impl.tuning.DataSplitter
import com.salesforce.op.utils.kryo.OpKryoRegistrator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

/**
 * Optimus Prime example classification app using the Titanic dataset
 */
object OpTitanic extends OpAppWithRunner with TitanicFeatures  {

  override def kryoRegistrator: Class[_ <: OpKryoRegistrator] = classOf[TitanicKryoRegistrator]

  ////////////////////////////////////////////////////////////////////////////////
  // READER DEFINITIONS
  /////////////////////////////////////////////////////////////////////////////////

  val randomSeed = 112233

  val trainingReader = new CSVReader[Passenger](
    readPath = None,
    schema = Passenger.getClassSchema.toString,
    key = _.getPassengerId.toString // entity to score
  ) {
    override def read(params: OpParams)
      (implicit sc: SparkSession): Either[RDD[Passenger], Dataset[Passenger]] = {
      val data = super.read(params)
      data match {
        case Left(dataRDD) =>
          val Array(train, _) = dataRDD.randomSplit(weights = Array(0.9, 0.1), seed = randomSeed)
          Left(train)
        case Right(dataSet) =>
          val Array(train, _) = dataSet.randomSplit(weights = Array(0.9, 0.1), seed = randomSeed)
          Right(train)
      }
    }
  }

  // Note not using the titanic test data because it has no labels so cannot evaluate
  val scoringReader = new CSVReader[Passenger](
    readPath = None,
    schema = Passenger.getClassSchema.toString,
    key = _.getPassengerId.toString // entity to score
  ) {
    override def read(params: OpParams)
      (implicit sc: SparkSession): Either[RDD[Passenger], Dataset[Passenger]] = {
      val data = super.read(params)
      data match {
        case Left(dataRDD) =>
          val Array(_, test) = dataRDD.randomSplit(weights = Array(0.9, 0.1), seed = randomSeed)
          Left(test)
        case Right(dataSet) =>
          val Array(_, test) = dataSet.randomSplit(weights = Array(0.9, 0.1), seed = randomSeed)
          Right(test)
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // WORKFLOW DEFINITION
  /////////////////////////////////////////////////////////////////////////////////

  val featureVector = Seq(pClass, name, sex, age, sibSp, parch, ticket, cabin, embarked).transmogrify()

  val checkedFeatures = new SanityChecker()
    .setCheckSample(0.10)
    .setSampleSeed(randomSeed)
    .setInput(survived, featureVector)
    .getOutput()


  val (pred, raw, prob) = BinaryClassificationModelSelector
    .withCrossValidation(splitter = Option(DataSplitter(seed = randomSeed)), seed = randomSeed)
    .setLogisticRegressionRegParam(0.05, 0.1)
    .setLogisticRegressionElasticNetParam(0.01)
    .setRandomForestMaxDepth(5, 10)
    .setRandomForestMinInstancesPerNode(10, 20, 30)
    .setRandomForestSeed(randomSeed)
    .setModelsToTry(LogisticRegression, RandomForest)
    .setInput(survived, checkedFeatures)
    .getOutput()

  val workflow = new OpWorkflow().setResultFeatures(pred, raw)

  val evaluator =
    Evaluators.BinaryClassification()
      .setLabelCol(survived).setPredictionCol(pred).setRawPredictionCol(raw)

  def runner(opParams: OpParams): OpWorkflowRunner =
    new OpWorkflowRunner(
      workflow = workflow,
      trainingReader = trainingReader,
      scoringReader = scoringReader,
      evaluationReader = Option(trainingReader),
      evaluator = evaluator,
      scoringEvaluator = None,
      featureToComputeUpTo = featureVector
    )

}
