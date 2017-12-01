/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import java.io.File

import com.github.fommil.netlib.{BLAS, LAPACK}
import com.salesforce.op.evaluators.{EvaluationMetrics, OpEvaluatorBase}
import com.salesforce.op.features.OPFeature
import com.salesforce.op.readers.Reader
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.OpSparkListener
import com.salesforce.op.utils.spark.RichRDD._
import enumeratum._
import org.apache.hadoop.io.compress.CompressionCodec
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.sql.SparkSession
import org.joda.time.Duration
import org.joda.time.format.PeriodFormatterBuilder
import org.scalacheck.Prop.Exception
import org.slf4j.LoggerFactory

import scala.util.{Failure, Success, Try}


/**
 * A class for running an Optimus Prime Workflow.
 * Provides methods to train, score, evaluate and computeUpTo for Optimus Prime Workflow.
 *
 * @param workflow             the workflow that you want to run (Note: the workflow should have the resultFeatures set)
 * @param trainingReader       reader to use to load up data for training
 * @param scoringReader        reader to use to load up data for scoring
 * @param evaluator            evaluator that you wish to use to evaluate the results of your workflow on a test dataset
 * @param scoringEvaluator     optional scoring evaluator that you wish to use when scoring
 * @param featureToComputeUpTo feature to generate data upto if calling the 'Features' run type
 */
class OpWorkflowRunner
(
  val workflow: OpWorkflow,
  val trainingReader: Reader[_],
  val scoringReader: Reader[_],
  val evaluationReader: Option[Reader[_]] = None,
  val evaluator: OpEvaluatorBase[_ <: EvaluationMetrics],
  val scoringEvaluator: Option[OpEvaluatorBase[_ <: EvaluationMetrics]] = None,
  val featureToComputeUpTo: OPFeature
) extends Serializable {

  @transient protected lazy val log = LoggerFactory.getLogger(this.getClass)

  /**
   * This method is called to train your workflow
   *
   * @param params parameters injected at runtime
   * @param spark  spark context which runs the workflow
   * @return TrainResult
   */
  protected def train(params: OpParams)(implicit spark: SparkSession): OpWorkflowRunnerResult = {
    val workflowModel = workflow.setReader(trainingReader).train()
    workflowModel.save(params.modelLocation.get)

    val modelSummary = workflowModel.summary()

    for {
      location <- params.metricsLocation
      metrics = spark.sparkContext.parallelize(Seq(modelSummary), 1)
      jobConf = {
        val conf = new JobConf(spark.sparkContext.hadoopConfiguration)
        conf.set("mapred.output.compress", params.metricsCompress.getOrElse(false).toString)
        conf
      }
      metricCodecClass = params.metricsCodec.map(Class.forName(_).asInstanceOf[Class[_ <: CompressionCodec]])
    } metrics.saveAsTextFile(location, metricCodecClass, jobConf)

    new TrainResult(modelSummary)
  }

  /**
   * This method can be used to generate features from part of your workflow for exploration outside the app
   *
   * @param params parameters injected at runtime
   * @param spark  spark context which runs the workflow
   * @return FeaturesResult
   */
  protected def computeFeatures(params: OpParams)(implicit spark: SparkSession): OpWorkflowRunnerResult = {
    workflow.setReader(trainingReader).computeDataUpTo(featureToComputeUpTo, params.writeLocation.get)
    new FeaturesResult()
  }

  /**
   * This method is usd to load up a previously trained workflow and use it to score a new dataset
   *
   * @param params parameters injected at runtime
   * @param spark  spark context which runs the workflow
   * @return ScoreResult
   */
  protected def score(params: OpParams)(implicit spark: SparkSession): OpWorkflowRunnerResult = {
    val workflowModel =
      workflow.loadModel(params.modelLocation.get)
        .setReader(scoringReader)
        .setParameters(params)

    val metrics = scoringEvaluator match {
      case None => workflowModel.score(path = params.writeLocation)
        None
      case Some(e) => val (_, met) = workflowModel.scoreAndEvaluate(e,
        path = params.writeLocation, metricsPath = params.metricsLocation)
      Option(met)
    }

    new ScoreResult(metrics)
  }

  /**
   * This method is used to call the evaluator on a test set of data scored by the trained workflow
   *
   * @param params parameters injected at runtime
   * @param spark  spark context which runs the workflow
   * @return EvaluateResult
   */
  protected def evaluate(params: OpParams)(implicit spark: SparkSession): OpWorkflowRunnerResult = {
    require(evaluationReader.isDefined, "The evaluate method requires an evaluation reader to be specified")
    val workflowModel =
      workflow.loadModel(params.modelLocation.get)
        .setReader(evaluationReader.get)
        .setParameters(params)

    val metrics = workflowModel.evaluate(
      evaluator,
      metricsPath = params.metricsLocation,
      scoresPath = params.writeLocation
    )
    new EvaluateResult(metrics)
  }

  /**
   * Run using a specified runner config
   *
   * @param runType  workflow run type
   * @param opParams op params
   * @return runner result
   */
  def run(runType: OpWorkflowRunType, opParams: OpParams)(implicit spark: SparkSession): OpWorkflowRunnerResult = {
    if (log.isDebugEnabled) {
      log.debug("BLAS implementation provided by: {}", BLAS.getInstance().getClass.getName)
      log.debug("LAPACK implementation provided by: {}", LAPACK.getInstance().getClass.getName)
    }

    log.info("Assuming OP params:\n{}", opParams)
    workflow.setParameters(opParams)

    // TODO: remove listener on app completion (should be available in spark 2.2.x - SPARK-18975)
    spark.sparkContext.addSparkListener(
      new OpSparkListener(
        appName = spark.sparkContext.appName,
        appId = spark.sparkContext.applicationId,
        runType = runType.toString,
        customTagName = opParams.customTagName,
        customTagValue = opParams.customTagValue,
        logStageMetrics = opParams.logStageMetrics.getOrElse(false)
      )
    )

    val start = DateTimeUtils.now()
    val result = runType match {
      case OpWorkflowRunType.Train => train(opParams)
      case OpWorkflowRunType.Score => score(opParams)
      case OpWorkflowRunType.Features => computeFeatures(opParams)
      case OpWorkflowRunType.Evaluate => evaluate(opParams)
    }
    val duration = new Duration(DateTimeUtils.now().minus(start.getMillis).getMillis)
    val total = new PeriodFormatterBuilder()
      .appendHours().appendSuffix("h")
      .appendMinutes().appendSuffix("m")
      .appendSecondsWithOptionalMillis().appendSuffix("s")
      .toFormatter.print(duration.toPeriod())

    log.info("Total run time: {}", total)
    log.info(result.toString)
    result
  }
}


sealed trait OpWorkflowRunType extends EnumEntry with Serializable
object OpWorkflowRunType extends Enum[OpWorkflowRunType] {
  val values = findValues
  case object Train extends OpWorkflowRunType
  case object Score extends OpWorkflowRunType
  case object Features extends OpWorkflowRunType
  case object Evaluate extends OpWorkflowRunType
}

/**
 * OpWorkflowRunner configuration container
 *
 * @param runType         workflow run type
 * @param paramLocation   workflow file params location
 * @param defaultParams   default params to use in case the file params is missing
 * @param readLocations   read locations
 * @param writeLocation   write location
 * @param modelLocation   model location
 * @param metricsLocation metrics location
 */
case class OpWorkflowRunnerConfig
(
  runType: OpWorkflowRunType = null,
  paramLocation: Option[String] = None,
  defaultParams: OpParams = new OpParams(),
  readLocations: Map[String, String] = Map.empty,
  writeLocation: Option[String] = None,
  modelLocation: Option[String] = None,
  metricsLocation: Option[String] = None
) extends JsonLike {

  /**
   * Convert the runner config into OpParams
   *
   * @return OpParams
   */
  def toOpParams: Try[OpParams] = Try {
    // Load params from paramLocation if specified
    val params = paramLocation match {
      case None => defaultParams
      case Some(pl) =>
        OpParams.fromFile(new File(pl)) match {
          case Failure(e) => throw new IllegalArgumentException(s"Failed to parse OP params from $pl", e)
          case Success(p) => p
        }
    }
    // Command line arguments take precedence over params file values
    params.withValues(
      readLocations = readLocations,
      writeLocation = writeLocation,
      modelLocation = modelLocation,
      metricsLocation = metricsLocation
    )
  }

  /**
   * Validate the config
   *
   * @return either error or op params
   */
  def validate: Either[String, OpParams] =
    this.toOpParams match {
      case Failure(e) => Left(e.toString)
      case Success(opParams) => runType match {
        case OpWorkflowRunType.Train if opParams.modelLocation.isEmpty =>
          Left("Must provide location to store model when training")
        case OpWorkflowRunType.Score if opParams.modelLocation.isEmpty || opParams.writeLocation.isEmpty =>
          Left("Must provide locations to read model and write data when scoring")
        case OpWorkflowRunType.Features if opParams.writeLocation.isEmpty =>
          Left("Must provide location to write data when generating features")
        case OpWorkflowRunType.Evaluate if opParams.modelLocation.isEmpty || opParams.metricsLocation.isEmpty =>
          Left("Must provide locations to read model and write metrics when evaluating")
        case _ => Right(opParams)
      }
    }

}

trait OpWorkflowRunnerResult extends Serializable
class TrainResult(val modelSummary: String) extends OpWorkflowRunnerResult {
  override def toString: String = s"Train result: $modelSummary"
}
class ScoreResult(val metrics: Option[EvaluationMetrics]) extends OpWorkflowRunnerResult {
  override def toString: String = s"Score result: ${metrics.getOrElse("{}")}"
}
class FeaturesResult() extends OpWorkflowRunnerResult {
  override def toString: String = s"Features result: {}"
}
class EvaluateResult(val metrics: EvaluationMetrics) extends OpWorkflowRunnerResult {
  override def toString: String = s"Evaluation result: $metrics"
}
