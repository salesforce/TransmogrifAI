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

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.evaluators.{OpBinaryClassificationEvaluatorBase, OpEvaluatorBase, OpMultiClassificationEvaluatorBase, SingleMetric}
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.features.{Feature, FeatureBuilder}
import com.salesforce.op.stages.OpPipelineStage2
import com.salesforce.op.stages.impl.selector.{ModelSelectorBaseNames, StageParamNames, _}
import com.salesforce.op.utils.spark.RichParamMap._
import com.salesforce.op.utils.stages.FitStagesUtil
import com.salesforce.op.utils.stages.FitStagesUtil._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.{Dataset, Row, SparkSession, functions}
import org.apache.spark.util.SparkThreadUtils
import org.slf4j.{Logger, LoggerFactory}

import scala.concurrent.duration.Duration
import scala.concurrent.{ExecutionContext, Future}


/**
 * Best Estimator container
 *
 * @param name      the name of the best model
 * @param estimator best estimator
 * @param summary  optional metadata
 * @tparam E model type
 */
case class BestEstimator[E <: Estimator[_]](name: String, estimator: E, summary: Seq[ModelEvaluation])

/**
 * Validated Model container
 *
 * @param model     model instance
 * @param bestIndex best metric / grid index
 * @param metrics   all computed metrics
 * @param grids     all param grids
 */
private[tuning] case class ValidatedModel[E <: Estimator[_]]
(
  model: E,
  bestIndex: Int,
  metrics: Array[Double],
  grids: Array[ParamMap]
) {
  /**
   * Best metric (metric at bestIndex)
   */
  def bestMetric: Double = metrics(bestIndex)

  /**
   * Best grid (param grid at bestIndex)
   */
  def bestGrid: ParamMap = grids(bestIndex)
}

/**
 * Abstract class for Validator: Cross Validation or Train Validation Split
 * The model type should be the output of the estimator type but specifying that kills the scala compiler
 */
private[op] trait OpValidator[M <: Model[_], E <: Estimator[_]] extends Serializable {

  @transient protected lazy val log: Logger = LoggerFactory.getLogger(this.getClass)

  def seed: Long

  def evaluator: OpEvaluatorBase[_]

  def validationName: String

  def stratify: Boolean

  def parallelism: Int

  private[op] final def isClassification = evaluator match {
    case _: OpBinaryClassificationEvaluatorBase[_] => true
    case _: OpMultiClassificationEvaluatorBase[_] => true
    case _ => false
  }

  def getParams(): Map[String, Any]

  /**
   * Function that performs the model selection
   *
   * @param modelInfo
   * @param dataset
   * @param label
   * @param features
   * @param dag
   * @param splitter
   * @param stratifyCondition Condition to stratify CV/TS
   * @param spark
   * @return
   */
  private[op] def validate[T](
    modelInfo: Seq[(E, Array[ParamMap])],
    dataset: Dataset[T],
    label: String,
    features: String,
    dag: Option[StagesDAG] = None,
    splitter: Option[Splitter] = None,
    stratifyCondition: Boolean = isClassification && stratify
  )(implicit spark: SparkSession): BestEstimator[E]


  /**
   * Get the best model and the metadata with with the validator params
   *
   * @param modelsFit     info from validation
   * @param bestEstimator best fit model
   * @param splitInfo     split info for logging
   * @return best model
   */
  private[op] def wrapBestEstimator(
    modelsFit: Array[ValidatedModel[E]],
    bestEstimator: E,
    splitInfo: String
  ): BestEstimator[E] = {
    log.info(
      "Model Selection over {} with {} with {} and the {} metric",
      modelsFit.map(_.model.getClass.getSimpleName).mkString(","), validationName, splitInfo, evaluator.name
    )
    val modelSummaries = modelsFit.flatMap(v => makeModelSummary(v))
    val bestModelName =
      if (evaluator.isLargerBetter) modelSummaries.maxBy(_.metricValues.asInstanceOf[SingleMetric].value).modelName
      else modelSummaries.minBy(_.metricValues.asInstanceOf[SingleMetric].value).modelName

    BestEstimator(name = bestModelName, estimator = bestEstimator, summary = modelSummaries)
  }

  /**
   * Update metadata during model selection and return best model name
   *
   * @return best model name
   */
  private[op] def makeModelSummary(v: ValidatedModel[E]): Seq[ModelEvaluation] = {
    val ValidatedModel(model, _, metrics, grids) = v
    val modelParams = model.extractParamMap().getAsMap()

    def makeModelName(index: Int) = s"${model.uid}_$index"

    for {((paramGrid, met), ind) <- grids.zip(metrics).zipWithIndex} yield {
      val updatedParams = modelParams ++ paramGrid.getAsMap()
      ModelEvaluation(
        modelUID = model.uid,
        modelName = makeModelName(ind),
        modelType = model.getClass.getSimpleName,
        metricValues = SingleMetric(evaluator.name.humanFriendlyName, met),
        modelParameters = updatedParams
      )
    }
  }


  /**
   * Creates Train Validation Splits
   *
   * @param stratifyCondition condition to stratify splits
   * @param dataset
   * @param label
   * @param splitter  used to estimate splitter params prior to splits
   * @return
   */
  private[op] def createTrainValidationSplits[T](stratifyCondition: Boolean,
    dataset: Dataset[T], label: String, splitter: Option[Splitter] = None): Array[(RDD[Row], RDD[Row])]


  protected def prepareStratification[T](
    dataset: Dataset[T],
    message: String,
    label: String,
    splitter: Option[Splitter] = None
  ): Array[RDD[Row]] = {
    log.info(message)
    import dataset.sqlContext.implicits._
    val classes = dataset.select(label).as[Double].distinct().collect().sorted
    val datasetsByClass = classes.map(theClass => dataset.filter(functions.col(label) === theClass))

    splitter.map {
      case d: DataBalancer => {
        val Array(negative, positive) = datasetsByClass
        d.estimate(
          data = dataset,
          positiveData = positive,
          negativeData = negative,
          seed = d.getSeed
        )
      }
      case c: DataCutter => {
        val labelCounts = dataset.sparkSession.createDataFrame(classes zip datasetsByClass.map(_.count())).persist
        c.estimate(labelCounts)
        labelCounts.unpersist
      }
      case _ =>
    }
    // Creates RDD grouped by classes (0, 1, 2, 3, ..., K)
    datasetsByClass.map(_.toDF().rdd)
  }

  protected def applyDAG(
    dag: StagesDAG,
    training: Dataset[Row],
    validation: Dataset[Row],
    label: String,
    features: String,
    splitter: Option[Splitter]
  )(implicit sparkSession: SparkSession): (Dataset[Row], Dataset[Row]) = {

    val FittedDAG(newTrain, newTest, _) = FitStagesUtil.fitAndTransformDAG(
      dag = dag,
      train = training,
      test = validation,
      hasTest = true,
      indexOfLastEstimator = Some(-1)
    )
    val selectTrain = newTrain.select(label, features)
      .withColumn(ModelSelectorBaseNames.idColName, monotonically_increasing_id())

    val selectTest = newTest.select(label, features)
      .withColumn(ModelSelectorBaseNames.idColName, monotonically_increasing_id())

    val (balancedTrain, balancedTest) = splitter.map(s => (
      s.prepare(selectTrain).train,
      s.prepare(selectTest).train)
    ).getOrElse((selectTrain, selectTest))

    (balancedTrain, balancedTest)
  }

  /**
   * Suppress logging to a specified level when executing method `f`.
   */
  protected def suppressLoggingForFun[Result](level: Level = Level.ERROR)(f: => Result): Result = {
    val opLog = LogManager.getLogger("com.salesforce.op")
    val originalLevel = opLog.getLevel
    opLog.setLevel(level)
    val result = f
    opLog.setLevel(originalLevel) // Reset log level back to normal
    result
  }

  protected def getSummary[T](
    modelInfo: Seq[(E, Array[ParamMap])], label: String, features: String, train: Dataset[T], test: Dataset[T]
  )(implicit ec: ExecutionContext): Array[ValidatedModel[E]] = {
    train.persist()
    test.persist()
    val summaryFuts = modelInfo.map { case (estimator, params) =>
      val name = estimator.getClass.getSimpleName
      estimator match {
        case e: OpPipelineStage2[RealNN, OPVector, Prediction]@unchecked =>
          val (labelFeat, Array(featuresFeat: Feature[OPVector]@unchecked, _)) =
            FeatureBuilder.fromDataFrame[RealNN](train.toDF(), response = label,
              nonNullable = Set(features, ModelSelectorBaseNames.idColName))
          e.setInput(labelFeat, featuresFeat)
          evaluator.setFullPredictionCol(e.getOutput())
        case _ => // otherwise it is a spark estimator
          val pi1 = estimator.getParam(StageParamNames.inputParam1Name)
          val pi2 = estimator.getParam(StageParamNames.inputParam2Name)
          estimator.set(pi1, label).set(pi2, features)
      }
      Future {
        val numModels = params.length
        val metrics = new Array[Double](params.length)
        log.info(s"Train split with multiple sets of parameters.")
        val models = estimator.fit(train, params).asInstanceOf[Seq[M]]
        for {i <- 0 until numModels} {
          val metric = evaluator.evaluate(models(i).transform(test, params(i)))
          log.info(s"Got metric $metric for model $name trained with ${params(i)}.")
          metrics(i) = metric
        }
        val (bestMetric, bestIndex) =
          if (evaluator.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
          else metrics.zipWithIndex.minBy(_._1)

        log.info(s"Best set of parameters:\n${params(bestIndex)} for $name")
        log.info(s"Best train validation split metric: $bestMetric.")
        ValidatedModel(estimator, bestIndex, metrics, params)
      }
    }
    val summary = SparkThreadUtils.utils.awaitResult(Future.sequence(summaryFuts), Duration.Inf).toArray
    train.unpersist()
    test.unpersist()
    summary
  }

  protected def getValidatedModel(modelSummaries: Array[ValidatedModel[E]]): ValidatedModel[E] = {
    if (evaluator.isLargerBetter) modelSummaries.maxBy(_.bestMetric) else modelSummaries.minBy(_.bestMetric)
  }

  protected def makeExecutionContext(numOfThreads: Int = parallelism): ExecutionContext = {
    if (numOfThreads <= 1) SparkThreadUtils.utils.sameThread
    else ExecutionContext.fromExecutorService(
      SparkThreadUtils.utils.newDaemonCachedThreadPool(s"${this.getClass.getSimpleName}-thread-pool", numOfThreads))
  }

}

object ValidatorParamDefaults {
  def Seed: Long = util.Random.nextLong // scalastyle:off method.name
  val LabelCol = "labelCol"
  val NumFolds = 3
  val TrainRatio = 0.75
  val Stratify = false
  val Parallelism = 8
}

