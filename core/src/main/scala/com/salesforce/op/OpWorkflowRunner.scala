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

package com.salesforce.op

import java.io.File

import com.github.fommil.netlib.{BLAS, LAPACK}
import com.salesforce.op.evaluators.{EvaluationMetrics, OpEvaluatorBase}
import com.salesforce.op.features.OPFeature
import com.salesforce.op.readers.{Reader, StreamingReader}
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.json.{EnumEntrySerializer, JsonLike, JsonUtils}
import com.salesforce.op.utils.spark.RichRDD._
import com.salesforce.op.utils.spark.{AppMetrics, OpSparkListener}
import com.salesforce.op.utils.version.VersionInfo
import enumeratum._
import org.apache.hadoop.io.compress.CompressionCodec
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.scheduler.{SparkListener, SparkListenerApplicationEnd}
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.StreamingContext
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.{Failure, Success, Try}

/**
 * A class for running an TransmogrifAI Workflow.
 * Provides methods to train, score, evaluate and computeUpTo for TransmogrifAI Workflow.
 *
 * @param workflow             the workflow that you want to run (Note: the workflow should have the resultFeatures set)
 * @param trainingReader       reader to use to load up data for training
 * @param scoringReader        reader to use to load up data for scoring
 * @param evaluationReader     reader to use to load up data for evaluation
 * @param streamingScoreReader reader to use to load up data for streaming score
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
  val streamingScoreReader: Option[StreamingReader[_]] = None,
  val evaluator: Option[OpEvaluatorBase[_ <: EvaluationMetrics]] = None,
  val scoringEvaluator: Option[OpEvaluatorBase[_ <: EvaluationMetrics]] = None,
  val featureToComputeUpTo: Option[OPFeature] = None
) extends Serializable {

  /**
   * OpWorkflowRunner ctor
   *
   * @param workflow             the workflow that you want to run
   *                             (Note: the workflow should have the resultFeatures set)
   * @param trainingReader       reader to use to load up data for training
   * @param scoringReader        reader to use to load up data for scoring
   * @param evaluationReader     reader to use to load up data for evaluation
   * @param evaluator            evaluator that you wish to use to evaluate
   *                             the results of your workflow on a test dataset
   * @param scoringEvaluator     optional scoring evaluator that you wish to use when scoring
   * @param featureToComputeUpTo feature to generate data upto if calling the 'Features' run type
   * @return OpWorkflowRunner
   */
  @deprecated("Use alternative ctor", "3.2.3")
  def this(
    workflow: OpWorkflow,
    trainingReader: Reader[_],
    scoringReader: Reader[_],
    evaluationReader: Option[Reader[_]],
    evaluator: OpEvaluatorBase[_ <: EvaluationMetrics],
    scoringEvaluator: Option[OpEvaluatorBase[_ <: EvaluationMetrics]],
    featureToComputeUpTo: OPFeature
  ) = this(
    workflow = workflow, trainingReader = trainingReader, scoringReader = scoringReader,
    evaluationReader = evaluationReader, streamingScoreReader = None, evaluator = Option(evaluator),
    scoringEvaluator = scoringEvaluator, featureToComputeUpTo = Option(featureToComputeUpTo)
  )

  /**
   * OpWorkflowRunner ctor
   *
   * @param workflow             the workflow that you want to run
   *                             (Note: the workflow should have the resultFeatures set)
   * @param trainingReader       reader to use to load up data for training
   * @param scoringReader        reader to use to load up data for scoring
   * @param evaluator            evaluator that you wish to use to evaluate
   *                             the results of your workflow on a test dataset
   * @param featureToComputeUpTo feature to generate data upto if calling the 'Features' run type
   * @return OpWorkflowRunner
   */
  @deprecated("Use alternative ctor", "3.2.3")
  def this(
    workflow: OpWorkflow,
    trainingReader: Reader[_],
    scoringReader: Reader[_],
    evaluator: OpEvaluatorBase[_ <: EvaluationMetrics],
    featureToComputeUpTo: OPFeature
  ) = this(
    workflow = workflow, trainingReader = trainingReader, scoringReader = scoringReader, evaluationReader = None,
    streamingScoreReader = None, evaluator = Option(evaluator), scoringEvaluator = None,
    featureToComputeUpTo = Option(featureToComputeUpTo)
  )

  @transient protected lazy val log = LoggerFactory.getLogger(classOf[OpWorkflowRunner])

  // Handles collecting the metrics on app completion
  private val onAppEndHandlers = ArrayBuffer.empty[AppMetrics => Unit]

  /**
   * Add handle to collect the app metrics when the application ends
   * @param h a handle to collect the app metrics when the application ends
   */
  final def addApplicationEndHandler(h: AppMetrics => Unit): this.type = {
    onAppEndHandlers += h
    this
  }

  private def onApplicationEnd(metrics: AppMetrics): Unit = {
    onAppEndHandlers.foreach(h =>
      Try(h(metrics)).recover { case e => log.error("Failed to handle application end event", e) }
    )
  }

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
    require(featureToComputeUpTo.isDefined,
      "The computeFeatures method requires featureToComputeUpTo to be specified")
    workflow.setReader(trainingReader).computeDataUpTo(featureToComputeUpTo.get, params.writeLocation.get)
    new FeaturesResult()
  }

  /**
   * This method is used to load up a previously trained workflow and use it to score a new dataset
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
      case None =>
        workflowModel.score(path = params.writeLocation)
        None
      case Some(e) =>
        val (_, metrcs) = workflowModel.scoreAndEvaluate(
          evaluator = e, path = params.writeLocation, metricsPath = params.metricsLocation
        )
        Option(metrcs)
    }

    new ScoreResult(metrics)
  }

  /**
   * This method is used to load up a previously trained workflow and use it to stream scores to a write location
   *
   * @param params    parameters injected at runtime
   * @param spark     spark context which runs the workflow
   * @param streaming spark streaming context which runs the workflow
   * @return StreamingScoreResult
   */
  protected def streamingScore(params: OpParams)
    (implicit spark: SparkSession, streaming: StreamingContext): OpWorkflowRunnerResult = {
    require(streamingScoreReader.isDefined,
      "The streamingScore method requires an streaming score reader to be specified")

    // Prepare write path
    def writePath(timeInMs: Long) = Some(s"${params.writeLocation.get.stripSuffix("/")}/$timeInMs")

    // Load the model to score with and prepare the scoring function
    val workflowModel = workflow.loadModel(params.modelLocation.get).setParameters(params)
    val scoreFn: Option[String] => _ = workflowModel.scoreFn(
      keepRawFeatures = false, keepIntermediateFeatures = false, persistScores = false
    )

    // Get the streaming score reader and create input stream
    val reader = streamingScoreReader.get.asInstanceOf[StreamingReader[Any]]
    val inputStream = reader.stream(params)

    inputStream.foreachRDD(rdd => {
      // Only score non empty datasets
      if (!rdd.isEmpty()) {
        val start = DateTimeUtils.now().getMillis
        log.info("Scoring a records batch")
        // Set input rdd for the workflow to score
        workflowModel.setInputRDD[Any](rdd, reader.key)(reader.wtt.asInstanceOf[WeakTypeTag[Any]])
        val path = writePath(start) // Prepare write path
        scoreFn(path) // Score & save it
        log.info("Scored a records batch in {}ms. Saved scores to {}", DateTimeUtils.now().getMillis - start, path)
      }
    })
    new StreamingScoreResult()
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
    require(evaluator.isDefined, "The evaluate method requires an evaluator to be specified")

    val workflowModel =
      workflow.loadModel(params.modelLocation.get)
        .setReader(evaluationReader.get)
        .setParameters(params)

    val metrics = workflowModel.evaluate(
      evaluator = evaluator.get, metricsPath = params.metricsLocation, scoresPath = params.writeLocation
    )
    new EvaluateResult(metrics)
  }

  /**
   * Run using a specified runner config
   *
   * @param runType   workflow run type
   * @param opParams  parameters injected at runtime
   * @param spark     spark context which runs the workflow
   * @param streaming spark streaming context which runs the workflow
   * @return runner result
   */
  def run(runType: OpWorkflowRunType, opParams: OpParams)
    (implicit spark: SparkSession, streaming: StreamingContext): OpWorkflowRunnerResult = {
    log.info("OP version info:\n{}", VersionInfo().toJson())
    log.info("Assuming OP params:\n{}", opParams)
    workflow.setParameters(opParams)

    log.info("BLAS implementation provided by: {}", BLAS.getInstance().getClass.getName)
    log.info("LAPACK implementation provided by: {}", LAPACK.getInstance().getClass.getName)

    val listener = sparkListener(runType, opParams)
    spark.sparkContext.addSparkListener(listener)

    val result = runType match {
      case OpWorkflowRunType.Train => train(opParams)
      case OpWorkflowRunType.Score => score(opParams)
      case OpWorkflowRunType.Features => computeFeatures(opParams)
      case OpWorkflowRunType.Evaluate => evaluate(opParams)
      case OpWorkflowRunType.StreamingScore =>
        val res = streamingScore(opParams)
        val timeoutInMs = opParams.awaitTerminationTimeoutSecs.map(_ * 1000L).getOrElse(-1L)
        try {
          streaming.start()
          streaming.awaitTerminationOrTimeout(timeoutInMs)
        } catch { case e: Exception => log.error("Streaming context error: ", e) }
        res
    }
    log.info(result.toString)
    result
  }

  private def sparkListener(
    runType: OpWorkflowRunType,
    opParams: OpParams
  )(implicit spark: SparkSession): SparkListener = {

    val collectStageMetrics = opParams.collectStageMetrics.exists {
      case true if runType == OpWorkflowRunType.StreamingScore =>
        log.warn("Stage metrics collection is not available in streaming context")
        false
      case v => v
    }
    new OpSparkListener(
      appName = spark.sparkContext.appName,
      appId = spark.sparkContext.applicationId,
      runType = runType.toString,
      customTagName = opParams.customTagName,
      customTagValue = opParams.customTagValue,
      logStageMetrics = opParams.logStageMetrics.getOrElse(false),
      collectStageMetrics = collectStageMetrics
    ) {
      override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {
        super.onApplicationEnd(applicationEnd)
        val m = super.metrics
        log.info("Total run time: {}", m.appDurationPretty)
        OpWorkflowRunner.this.onApplicationEnd(m)
        spark.sparkContext.removeSparkListener(this)
      }
    }
  }

}


sealed trait OpWorkflowRunType extends EnumEntry with Serializable
object OpWorkflowRunType extends Enum[OpWorkflowRunType] {
  val values = findValues
  case object Train extends OpWorkflowRunType
  case object Score extends OpWorkflowRunType
  case object StreamingScore extends OpWorkflowRunType
  case object Features extends OpWorkflowRunType
  case object Evaluate extends OpWorkflowRunType
}

/**
 * OpWorkflowRunner configuration container
 *
 * @param runType                 workflow run type
 * @param paramLocation           workflow file params location
 * @param defaultParams           default params to use in case the file params is missing
 * @param readLocations           read locations
 * @param writeLocation           write location
 * @param modelLocation           model location
 * @param metricsLocation         metrics location
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

  override def toJson(pretty: Boolean = true): String = {
    val serdes = EnumEntrySerializer.jackson[OpWorkflowRunType](OpWorkflowRunType)
    JsonUtils.toJsonString(this, pretty = pretty, serdes = Seq(serdes))
  }

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
        case OpWorkflowRunType.StreamingScore if opParams.modelLocation.isEmpty || opParams.writeLocation.isEmpty =>
          Left("Must provide locations to read model and write data when streaming score")
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
class StreamingScoreResult() extends OpWorkflowRunnerResult {
  override def toString: String = "Streaming Score result: {}"
}
class FeaturesResult() extends OpWorkflowRunnerResult {
  override def toString: String = "Features result: {}"
}
class EvaluateResult(val metrics: EvaluationMetrics) extends OpWorkflowRunnerResult {
  override def toString: String = s"Evaluation result: $metrics"
}
