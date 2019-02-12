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

import com.salesforce.op.evaluators.OpEvaluatorBase
import com.salesforce.op.stages.impl.selector.ModelSelectorNames
import com.salesforce.op.utils.stages.FitStagesUtil._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}

import scala.concurrent.ExecutionContext


private[op] class OpTrainValidationSplit[M <: Model[_], E <: Estimator[_]]
(
  val trainRatio: Double = ValidatorParamDefaults.TrainRatio,
  val seed: Long = ValidatorParamDefaults.Seed,
  val evaluator: OpEvaluatorBase[_],
  val stratify: Boolean = ValidatorParamDefaults.Stratify,
  val parallelism: Int = ValidatorParamDefaults.Parallelism
) extends OpValidator[M, E] {

  val validationName: String = ModelSelectorNames.TrainValSplitResults

  override def getParams(): Map[String, Any] = Map("trainRatio" -> trainRatio, "seed" -> seed,
    "evaluator" -> evaluator.name.humanFriendlyName, "stratify" -> stratify, "parallelism" -> parallelism)

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

    val (training, validation) = createTrainValidationSplits(
      stratifyCondition = stratifyCondition,
      dataset = dataset,
      label = label,
      splitter = splitter
    ).head

    val trainingDataset = dataset.sparkSession.createDataFrame(training, schema)
    val validationDataset = dataset.sparkSession.createDataFrame(validation, schema)

    // If there is a TS DAG, then run it
    val (newTrain, newTest) = suppressLoggingForFun() {
      dag.map(theDAG => applyDAG(
        dag = theDAG,
        training = trainingDataset,
        validation = validationDataset,
        label = label,
        features = features,
        splitter = splitter
      )).getOrElse(trainingDataset, validationDataset)
    }
    implicit val ec: ExecutionContext = makeExecutionContext()
    val modelSummaries = getSummary(modelInfo, label = label, features = features, train = newTrain, test = newTest)
    dataset.unpersist()

    val model = getValidatedModel(modelSummaries)
    wrapBestEstimator(modelSummaries, model.model.copy(model.bestGrid).asInstanceOf[E], s"$trainRatio training split")
  }

  /**
   * Creates Train Validation Splits For TS
   *
   * @param stratifyCondition condition to do stratify ts
   * @param dataset           dataset to split
   * @param label             name of label in dataset
   * @param splitter          used to estimate splitter params prior to ts
   * @return Array[(Train, Test)]
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

    val Array(train, test) = if (stratifyCondition) {
      val rddsByClass = prepareStratification(
        dataset = dataset,
        message = s"Creating stratified train/validation with training ratio of $trainRatio",
        label = label,
        splitter = splitter
      )
      stratifyTrainValidationSplit(rddsByClass)
    } else {
      val rddRow = dataset.toDF().rdd
      rddRow.randomSplit(Array(trainRatio, 1 - trainRatio), seed)
    }
    Array((train, test))
  }

  private def stratifyTrainValidationSplit(rddsByClass: Array[RDD[Row]]): Array[RDD[Row]] = {
    // Train/Validation data for each class
    val splitByClass = rddsByClass.map(_.randomSplit(Array(trainRatio, 1 - trainRatio), seed))

    if (splitByClass.isEmpty) throw new Error("Train Validation Data Grouped by class is empty")
    // Merging Train/Validation data one by one
    splitByClass.reduce[Array[RDD[Row]]] {
      case (Array(train1: RDD[Row], validate1: RDD[Row]), Array(train2: RDD[Row], validate2: RDD[Row])) =>
        Array(train1.union(train2), validate1.union(validate2))
    }
  }

}

