/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.utils.stages.FitStagesUtil._
import com.salesforce.op.utils.stages.FitStagesUtil
import com.salesforce.op.evaluators.{OpBinaryClassificationEvaluatorBase, OpEvaluatorBase, OpMultiClassificationEvaluatorBase}
import com.salesforce.op.stages.impl.selector.{ModelInfo, ModelSelectorBaseNames, StageParamNames}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.types.{MetadataBuilder, StructType}
import org.apache.spark.sql.{Dataset, Row, SparkSession, functions}
import org.slf4j.{Logger, LoggerFactory}


/**
 * Best Model container
 *
 * @param name     the name of the best model
 * @param model    best trained model
 * @param metadata optional metadata
 * @tparam M model type
 */
case class BestModel[M <: Model[_]](name: String, model: M, metadata: Option[MetadataBuilder] = None)

/**
 * Best Estimator container
 *
 * @param name      the name of the best model
 * @param estimator best estimator
 * @param metadata  optional metadata
 * @tparam E model type
 */
case class BestEstimator[E <: Estimator[_]](name: String, estimator: E, metadata: MetadataBuilder = new MetadataBuilder)

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
private[impl] trait OpValidator[M <: Model[_], E <: Estimator[_]] extends Serializable {

  @transient protected lazy val log: Logger = LoggerFactory.getLogger(this.getClass)

  type ModelWithGrids = Seq[(E, Array[ParamMap], String)]

  def seed: Long

  def evaluator: OpEvaluatorBase[_]

  def validationName: String

  def stratify: Boolean

  private[op] final def isClassification = evaluator match {
    case _: OpBinaryClassificationEvaluatorBase[_] => true
    case _: OpMultiClassificationEvaluatorBase[_] => true
    case _ => false
  }


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
    modelInfo: Seq[ModelInfo[E]],
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
    val meta = new MetadataBuilder()
    val cvFittedModels = modelsFit.map(v => updateBestModelMetadata(meta, v) -> v.bestMetric)
    val newMeta = new MetadataBuilder().putMetadata(validationName, meta.build())
    val (bestModelName, _) = if (evaluator.isLargerBetter) cvFittedModels.maxBy(_._2) else cvFittedModels.minBy(_._2)

    BestEstimator(name = bestModelName, estimator = bestEstimator, metadata = newMeta)
  }

  /**
   * Update metadata during model selection and return best model name
   *
   * @return best model name
   */
  private[op] def updateBestModelMetadata(metadataBuilder: MetadataBuilder, v: ValidatedModel[E]): String = {
    val ValidatedModel(model, bestIndex, metrics, grids) = v
    val modelParams = model.extractParamMap()

    def makeModelName(index: Int) = s"${model.uid}_$index"

    for {((paramGrid, met), ind) <- grids.zip(metrics).zipWithIndex} {
      val paramMetBuilder = new MetadataBuilder()
      paramMetBuilder.putString(evaluator.name, met.toString)
      // put in all model params from the winner
      modelParams.toSeq.foreach(p => paramMetBuilder.putString(p.param.name, p.value.toString))
      // override with param map values if they exists
      paramGrid.toSeq.foreach(p => paramMetBuilder.putString(p.param.name, p.value.toString))
      metadataBuilder.putMetadata(makeModelName(ind), paramMetBuilder.build())
    }
    makeModelName(bestIndex)
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
    import sparkSession.implicits._

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

  protected def getValidatedModel(groupedSummary: Array[ValidatedModel[E]]): ValidatedModel[E] = {
    if (evaluator.isLargerBetter) groupedSummary.maxBy(_.bestMetric) else groupedSummary.minBy(_.bestMetric)
  }

  protected def getSummary[T](
    modelsWithGrids: ModelWithGrids, label: String, features: String, train: Dataset[T], test: Dataset[T]
  ): Array[ValidatedModel[E]] = {
    train.persist()
    test.persist()
    val summary = modelsWithGrids.par.map {
      case (estimator, paramGrids, name) =>
        val pi1 = estimator.getParam(StageParamNames.inputParam1Name)
        val pi2 = estimator.getParam(StageParamNames.inputParam2Name)
        estimator.set(pi1, label).set(pi2, features)

        val numModels = paramGrids.length
        val metrics = new Array[Double](paramGrids.length)

        log.info(s"Train split with multiple sets of parameters.")
        val models = estimator.fit(train, paramGrids).asInstanceOf[Seq[M]]
        var i = 0
        while (i < numModels) {
          val metric = evaluator.evaluate(models(i).transform(test, paramGrids(i)))
          log.info(s"Got metric $metric for model $name trained with ${paramGrids(i)}.")
          metrics(i) = metric
          i += 1
        }
        val (bestMetric, bestIndex) =
          if (evaluator.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
          else metrics.zipWithIndex.minBy(_._1)
        log.info(s"Best set of parameters:\n${paramGrids(bestIndex)} for $name")
        log.info(s"Best train validation split metric: $bestMetric.")

        ValidatedModel(estimator, bestIndex, metrics, paramGrids)
    }.toArray
    train.unpersist()
    test.unpersist()
    summary
  }

}

object ValidatorParamDefaults {
  def Seed: Long = util.Random.nextLong // scalastyle:off method.name
  val LabelCol = "labelCol"
  val NumFolds = 3
  val TrainRatio = 0.75
  val Stratify = false
}

