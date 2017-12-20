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


object /* APP_NAME >> */ Simple extends OpApp with Features {

  ////////////////////////////////////////////////////////////////////////////////
  // READER DEFINITIONS
  /////////////////////////////////////////////////////////////////////////////////
  val schema = DataClass.getClassSchema

  trait CanRead {
    val isTest: Boolean

    protected def split(
      data: Either[RDD[DataClass], Dataset[DataClass]],
      weights: Array[Double] = Array(0.9, 0.1)
    ): Either[RDD[DataClass], Dataset[DataClass]] = data match {
      case Left(rdd) =>
        val Array(train, test) = rdd.randomSplit(weights, randomSeed)
        Left(if (isTest) test else train)
      case Right(ds) =>
        val Array(train, test) = ds.randomSplit(weights, randomSeed)
        Right(if (isTest) test else train)
    }
  }

  trait Training extends CanRead {
    val isTest = false
  }

  trait Scoring extends CanRead {
    val isTest = true
  }

  abstract class ReaderWithHeaders
    extends CSVAutoReader[DataClass](
      readPath = None,
      headers = Seq.empty,
      recordNamespace = schema.getNamespace,
      recordName = schema.getName,
      key = _.getSomeId.toString /* << KEY_FN */
    ) with CanRead {
    override def read(params: OpParams)(implicit spark: SparkSession): Either[RDD[DataClass], Dataset[DataClass]] = {
      split(super.read(params))
    }
  }

  abstract class ReaderWithNoHeaders
    extends CSVReader[DataClass](
      readPath = None,
      schema = schema.toString,
      key = _.getSomeId.toString  /* << KEY_FN */
    ) with CanRead {
    override def read(params: OpParams)(implicit spark: SparkSession): Either[RDD[DataClass], Dataset[DataClass]] = {
      split(super.read(params))
    }
  }

  abstract class SampleReader extends ReaderWithHeaders  /* << READER_CHOICE */

  val randomSeed = 42 /* << RANDOM_SEED */

  private def split(
    data: Either[RDD[DataClass], Dataset[DataClass]],
    isTest: Boolean,
    weights: Array[Double] = Array(0.9, 0.1)
  ): Either[RDD[DataClass], Dataset[DataClass]] = data match {
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
    Seq(pClass, name, sex, age, sibSp, parch, ticket, cabin, embarked).transmogrify() /* << FEATURE_LIST */

  val labelFixed =
    Seq(survived).transmogrify() /* << RESPONSE_FEATURE */
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
      trainingReader = new SampleReader with Training,
      scoringReader = new SampleReader with Scoring,
      evaluator = evaluator,
      scoringEvaluator = Some(evaluator),
      featureToComputeUpTo = featureVector
    ).run(runType, opParams)
  }

}
