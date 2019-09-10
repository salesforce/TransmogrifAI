// scalastyle:off header.matches
/*
 * Modifications: (c) 2017, Salesforce.com, Inc.
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.salesforce.op.stages.impl.tuning

import com.github.fommil.netlib.BLAS
import com.salesforce.op.evaluators.{EvaluationMetrics, OpEvaluatorBase}
import com.salesforce.op.stages.OPStage
import com.salesforce.op.stages.impl.selector.ModelSelectorNames
import com.salesforce.op.utils.stages.FitStagesUtil._
import com.twitter.algebird.Operators._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.util.SparkThreadUtils
import com.twitter.algebird._
import com.twitter.algebird.Operators._

import scala.concurrent.duration.Duration
import scala.concurrent.{ExecutionContext, Future}


private[op] class OpCrossValidation[M <: Model[_], E <: Estimator[_]]
(
  val numFolds: Int = ValidatorParamDefaults.NumFolds,
  val seed: Long = ValidatorParamDefaults.Seed,
  val evaluator: OpEvaluatorBase[_ <: EvaluationMetrics],
  val stratify: Boolean = ValidatorParamDefaults.Stratify,
  val parallelism: Int = ValidatorParamDefaults.Parallelism,
  val maxWait: Duration = ValidatorParamDefaults.MaxWait
) extends OpValidator[M, E] {

  val validationName: String = ModelSelectorNames.CrossValResults
  private val blas = BLAS.getInstance()

  override def getParams(): Map[String, Any] = Map("numFolds" -> numFolds, "seed" -> seed,
    "evaluator" -> evaluator.name.humanFriendlyName, "stratify" -> stratify, "parallelism" -> parallelism)

  /**
   * Should be called only on instances of the same model
   */
  private def findBestModel(
    folds: Seq[ValidatedModel[E]]
  ): ValidatedModel[E] = {

    val gridCounts = folds.flatMap(_.grids.map(_ -> 1)).sumByKey
    val (_, maxFolds) = gridCounts.maxBy{ case (_, count) => count }
    val gridsIn = gridCounts.filter{ case (_, foldCount) => foldCount == maxFolds }.keySet

    implicit val doubleSemigroup = Semigroup.from[Double](_ + _)
    implicit val mapDoubleMonoid = Monoid.mapMonoid[String, Double](doubleSemigroup)
    val gridMetrics = folds.flatMap{
      f => f.grids.zip(f.metrics).collect { case (pm, met) if gridsIn.contains(pm) => (pm, met / maxFolds) }
    }.sumByKey

    val ((bestGrid, bestMetric), bestIndex) =
      if (evaluator.isLargerBetter) gridMetrics.zipWithIndex.maxBy{ case ((_, metric), _) => metric}
      else gridMetrics.zipWithIndex.minBy{ case ((_, metric), _) => metric}

    val ValidatedModel(est, _, _, _) = folds.head
    log.info(s"Average cross-validation for $est metrics: {}", gridMetrics.mkString(","))
    log.info(s"Best set of parameters:\n$bestGrid")
    log.info(s"Best cross-validation metric: $bestMetric.")
    val (grid, metrics) = gridMetrics.unzip
    ValidatedModel(est, bestIndex, metrics.toArray, grid.toArray)
  }

  private[op] override def validate[T](
    modelInfo: Seq[(E, Array[ParamMap])],
    dataset: Dataset[T],
    label: String,
    features: String,
    dag: Option[StagesDAG] = None,
    splitter: Option[Splitter] = None,
    stratifyCondition: Boolean = isClassification && stratify
  )(implicit spark: SparkSession): BestEstimator[E] = {

    dataset.persist()
    val schema = dataset.schema

    // creating k train/validation data
    val splits: Array[(RDD[Row], RDD[Row])] = createTrainValidationSplits(
      stratifyCondition = stratifyCondition, dataset = dataset, label = label, splitter = splitter
    )

    // Prepare copies of the DAG for each CV split so we can parallelize it safely
    val splitsWithDags =
      splits.zipWithIndex.map { case (data, splitIndex) =>
        val dagCopy = dag.map(_.map { layer =>
          layer.map { case (stage, depth) => (stage.copy(ParamMap.empty): OPStage) -> depth }: Layer
        })
        (data, splitIndex, dagCopy)
      }

    // Evaluate all models in parallel
    implicit val ec: ExecutionContext = makeExecutionContext()
    val modelSummariesFuts = splitsWithDags.map { case ((trainingRDD, validationRDD), splitIndex, theDAG) =>
      log.info(s"Cross Validation $splitIndex with multiple sets of parameters.")
      Future {
        suppressLoggingForFun() {
          val training = spark.createDataFrame(trainingRDD, schema)
          val validation = spark.createDataFrame(validationRDD, schema)
          val (newTrain, newTest) = theDAG.map((d: StagesDAG) =>
            // If there is a CV DAG, then run it
            applyDAG(
              dag = d, training = training, validation = validation,
              label = label, features = features, splitter = splitter
            )
          ).getOrElse(training -> validation)
          getSummary(modelInfo = modelInfo, label = label, features = features, train = newTrain, test = newTest)
        }
      }
    }
    // Await for all the evaluations to complete
    val modelSummaries = SparkThreadUtils.utils.awaitResult(Future.sequence(modelSummariesFuts.toSeq), maxWait)

    // Find the best model & return it
    val groupedSummary = modelSummaries.flatten.groupBy(_.model).map { case (_, folds) => findBestModel(folds) }.toArray
    val model = getValidatedModel(groupedSummary)
    val bestEstimator = wrapBestEstimator(
      groupedSummary, model.model.copy(model.bestGrid).asInstanceOf[E], s"$numFolds folds"
    )
    dataset.unpersist()
    bestEstimator
  }

  /**
   * Creates Train Validation Splits For CV
   *
   * @param stratifyCondition condition to do stratify cv
   * @param dataset dataset to split
   * @param label name of label in data
   * @param splitter  used to estimate splitter params prior to cv
   * @return Array((TrainRDD, ValidationRDD), Index)
   */
  private[op] override def createTrainValidationSplits[T](
    stratifyCondition: Boolean,
    dataset: Dataset[T],
    label: String,
    splitter: Option[Splitter]
  ): Array[(RDD[Row], RDD[Row])] = {

    // get param that stores the label column
    val labelCol = evaluator.getParam(ValidatorParamDefaults.LabelCol)
    evaluator.set(labelCol, label)

    // creating k train/validation data
    if (stratifyCondition) {
      val rddsByClass = prepareStratification(
        dataset = dataset,
        message = s"Creating $numFolds stratified folds",
        label = label,
        splitter = splitter
      )
      stratifyKFolds(rddsByClass)
    } else {
      val rddRow = dataset.toDF().rdd
      MLUtils.kFold(rddRow, numFolds, seed)
    }
  }

  private def stratifyKFolds(rddsByClass: Array[RDD[Row]]): Array[(RDD[Row], RDD[Row])] = {
    // Cross Validation's Train/Validation data for each class
    val foldsByClass = rddsByClass.map(rdd => MLUtils.kFold(rdd, numFolds, seed)).toSeq

    if (foldsByClass.isEmpty) {
      throw new RuntimeException("Dataset is too small for CV folds selected some empty datasets are created")
    }
    // Merging Train/Validation data one by one
    foldsByClass.reduce[Array[(RDD[Row], RDD[Row])]] {
      // cv1 and cv2 are arrays of train/validation data
      case (cv1: Array[(RDD[Row], RDD[Row])], cv2: Array[(RDD[Row], RDD[Row])]) =>
        (cv1 zip cv2).map { // zip the two arrays and merge the tuples one by one
          case ((train1: RDD[Row], test1: RDD[Row]), (train2: RDD[Row], test2: RDD[Row])) =>
            (train1.union(train2), test1.union(test2))
        }
    }
  }

}
