/* setFileName APP_NAME */
/* replace DataClass SCHEMA_NAME */
package com.salesforce.app

import com.salesforce.app.schema.DataClass /* << SCHEMA_IMPORT */
import com.salesforce.op._
import com.salesforce.op.readers._
import com.salesforce.op.evaluators._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification._
import com.salesforce.op.stages.impl.preparators._
import com.salesforce.op.stages.impl.regression._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}


object /* APP_NAME >> */ Simple extends OpAppWithRunner with Features {

  val randomSeed = 42 /* << RANDOM_SEED */

  ////////////////////////////////////////////////////////////////////////////////
  // READER DEFINITIONS
  /////////////////////////////////////////////////////////////////////////////////
  val schema = DataClass.getClassSchema

  type Data = Either[RDD[DataClass], Dataset[DataClass]]

  trait TrainTestSplit {
    def isTrain: Boolean

    protected def split(data: Data, weights: Array[Double] = Array(0.9, 0.1)): Data = data match {
      case Left(rdd) =>
        val Array(train, test) = rdd.randomSplit(weights, randomSeed)
        Left(if (isTrain) train else test)
      case Right(ds) =>
        val Array(train, test) = ds.randomSplit(weights, randomSeed)
        Right(if (isTrain) train else test)
    }
  }

  abstract class ReaderWithHeaders
    extends CSVAutoReader[DataClass](
      readPath = None,
      headers = Seq.empty,
      recordNamespace = schema.getNamespace,
      recordName = schema.getName,
      key = _.getSomeId.toString /* << KEY_FN */
    ) with TrainTestSplit {
    override def read(params: OpParams)(implicit spark: SparkSession): Data = split(super.read(params))
  }

  abstract class ReaderWithNoHeaders
    extends CSVReader[DataClass](
      readPath = None,
      schema = schema.toString,
      key = _.getSomeId.toString /* << KEY_FN */
    ) with TrainTestSplit {
    override def read(params: OpParams)(implicit spark: SparkSession): Data = split(super.read(params))
  }

  class SampleReader(val isTrain: Boolean) extends ReaderWithHeaders /* << READER_CHOICE */


  ////////////////////////////////////////////////////////////////////////////////
  // WORKFLOW DEFINITION
  /////////////////////////////////////////////////////////////////////////////////

  val featureVector =
    Seq(pClass, name, sex, age, sibSp, parch, ticket, cabin, embarked).transmogrify() /* << FEATURE_LIST */

  val label =
    Seq(survived).transmogrify() /* << RESPONSE_FEATURE */
      .map[RealNN](_.value(0).toRealNN)

  val checkedFeatures = new SanityChecker()
    .setCheckSample(0.10)
    .setInput(label, featureVector)
    .getOutput()

  // BEGIN PROBLEM_KIND
  val (pred, raw, prob) = BinaryClassificationModelSelector()
    .setInput(label, checkedFeatures)
    .getOutput()
  // END PROBLEM_KIND

  val workflow = new OpWorkflow().setResultFeatures(pred)

  def runner(opParams: OpParams): OpWorkflowRunner =
    new OpWorkflowRunner(
      workflow = workflow,
      trainingReader = new SampleReader(isTrain = true),
      scoringReader = new SampleReader(isTrain = false),
      evaluationReader = Option(new SampleReader(isTrain = false)),
      evaluator = Option(evaluator),
      scoringEvaluator = Option(evaluator)
    )

}
