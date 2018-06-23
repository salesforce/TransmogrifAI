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
import com.salesforce.op.stages.impl.selector.{ModelInfo, ModelSelectorBaseNames}
import com.salesforce.op.utils.stages.FitStagesUtil._
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}


private[op] class OpCrossValidation[M <: Model[_], E <: Estimator[_]]
(
  val numFolds: Int = ValidatorParamDefaults.NumFolds,
  val seed: Long = ValidatorParamDefaults.Seed,
  val evaluator: OpEvaluatorBase[_],
  val stratify: Boolean = ValidatorParamDefaults.Stratify
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

  // TODO use futures to parallelize https://github.com/apache/spark/commit/16c4c03c71394ab30c8edaf4418973e1a2c5ebfe
  private[op] override def validate[T](
    modelInfo: Seq[ModelInfo[E]],
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
      stratifyCondition = stratifyCondition,
      dataset = dataset,
      label = label,
      splitter = splitter
    )

    val modelsWithGrids = modelInfo.map(m => (m.sparkEstimator, m.grid.build(), m.modelName))

    // TODO use futures to parallelize https://github.com/apache/spark/commit/16c4c03c71394ab30c8edaf4418973e1a2c5ebfe
    val groupedSummary = suppressLoggingForFun() {
      splits.zipWithIndex.flatMap {
        case ((training, validation), splitIndex) => {
          log.info(s"Cross Validation $splitIndex with multiple sets of parameters.")
          val trainingDataset = spark.createDataFrame(training, schema)
          val validationDataset = spark.createDataFrame(validation, schema)
          val (newTrain, newTest) = dag.map(theDAG =>
            // If there is a CV DAG, then run it
            applyDAG(
              dag = theDAG,
              training = trainingDataset,
              validation = validationDataset,
              label = label,
              features = features,
              splitter = splitter
            )
          ).getOrElse(trainingDataset, validationDataset)
          getSummary(modelsWithGrids = modelsWithGrids, label = label, features = features,
            train = newTrain, test = newTest)
        }
      }.groupBy(_.model).map{ case (_, folds) => findBestModel(folds) }.toArray
    }
    dataset.unpersist()

    val model = getValidatedModel(groupedSummary)
    wrapBestEstimator(groupedSummary, model.model.copy(model.bestGrid).asInstanceOf[E], s"$numFolds folds")
  }

  // TODO : Implement our own kFold method for better performance in a separate PR
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
