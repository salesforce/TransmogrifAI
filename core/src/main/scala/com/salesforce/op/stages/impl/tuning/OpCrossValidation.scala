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
import com.salesforce.op.evaluators.OpEvaluatorBase
import com.salesforce.op.stages.OPStage
import com.salesforce.op.stages.impl.selector.{ModelInfo, ModelSelectorBaseNames}
import com.salesforce.op.utils.stages.FitStagesUtil._
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.util.SparkThreadUtils

import scala.concurrent.duration.Duration
import scala.concurrent.{ExecutionContext, Future}


private[op] class OpCrossValidation[M <: Model[_], E <: Estimator[_]]
(
  val numFolds: Int = ValidatorParamDefaults.NumFolds,
  val seed: Long = ValidatorParamDefaults.Seed,
  val evaluator: OpEvaluatorBase[_],
  val stratify: Boolean = ValidatorParamDefaults.Stratify,
  val parallelism: Int = ValidatorParamDefaults.Parallelism
) extends OpValidator[M, E] {

  val validationName: String = ModelSelectorBaseNames.CrossValResults
  private val blas = BLAS.getInstance()

  private def findBestModel(
    folds: Seq[ValidatedModel[E]]
  ): ValidatedModel[E] = {
    val metrics = folds.map(_.metrics).reduce(_ + _)
    blas.dscal(metrics.length, 1.0 / numFolds, metrics, 1)
    val ValidatedModel(est, _, _, grid) = folds.head
    log.info(s"Average cross-validation for $est metrics: {}", metrics.toSeq.mkString(","))
    val (bestMetric, bestIndex) =
      if (evaluator.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    log.info(s"Best set of parameters:\n${grid(bestIndex)}")
    log.info(s"Best cross-validation metric: $bestMetric.")
    ValidatedModel(est, bestIndex, metrics, grid)
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
    val modelSummaries = SparkThreadUtils.utils.awaitResult(Future.sequence(modelSummariesFuts.toSeq), Duration.Inf)

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
  private[op] override def createTrainValidationSplits[T](stratifyCondition: Boolean,
    dataset: Dataset[T], label: String, splitter: Option[Splitter] = None): Array[(RDD[Row], RDD[Row])] = {

    // TODO : Implement our own kFold method for better performance in a separate PR

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
      throw new RuntimeException("Dataset is too small for CV forlds selected some empty datasets are created")
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
