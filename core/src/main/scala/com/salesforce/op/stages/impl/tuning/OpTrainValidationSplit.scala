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

import com.salesforce.op.evaluators.{OpBinaryClassificationEvaluatorBase, OpEvaluatorBase, OpMultiClassificationEvaluatorBase}
import com.salesforce.op.stages.impl.selector.{ModelInfo, ModelSelectorBaseNames, StageParamNames}
import com.salesforce.op.stages.impl.tuning.SelectorData.LabelFeaturesKey
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.StructType


private[impl] class OpTrainValidationSplit[M <: Model[_], E <: Estimator[_]]
(
  val trainRatio: Double = ValidatorParamDefaults.TrainRatio,
  val seed: Long = ValidatorParamDefaults.Seed,
  val evaluator: OpEvaluatorBase[_],
  val stratify: Boolean = ValidatorParamDefaults.Stratify
) extends OpValidator[M, E] {

  val validationName: String = ModelSelectorBaseNames.TrainValSplitResults

  private[op] def validate(
    modelInfo: Seq[ModelInfo[E]],
    dataset: Dataset[_],
    label: String,
    features: String
  ): BestModel[M] = {
    // get param that stores the label column
    val labelCol = evaluator.getParam(ValidatorParamDefaults.labelCol)
    evaluator.set(labelCol, label)

    val schema = dataset.schema
    import dataset.sparkSession.implicits._
    val rdd = dataset.as[LabelFeaturesKey].rdd.persist()

    val (trainingRDD, validationRDD) = createTrainValidationSplits(rdd).head

    val sparkSession = dataset.sparkSession
    val newSchema = StructType(schema.dropRight(1)) // dropping key
    val trainingDataset = sparkSession.createDataFrame(trainingRDD, newSchema).persist()
    val validationDataset = sparkSession.createDataFrame(validationRDD, newSchema).persist()

    // multi-model training
    val modelWithGrid = modelInfo.map(m => (m.sparkEstimator, m.grid.build(), m.modelName))
    val groupedSummary = modelWithGrid.par.map {
      case (estimator, paramGrids, name) =>
        val pi1 = estimator.getParam(StageParamNames.inputParam1Name)
        val pi2 = estimator.getParam(StageParamNames.inputParam2Name)
        estimator.set(pi1, label).set(pi2, features)

        val numModels = paramGrids.length
        val metrics = new Array[Double](paramGrids.length)

        log.info(s"Train split with multiple sets of parameters.")
        val models = estimator.fit(trainingDataset, paramGrids).asInstanceOf[Seq[M]]
        var i = 0
        while (i < numModels) {
          val metric = evaluator.evaluate(models(i).transform(validationDataset, paramGrids(i)))
          log.info(s"Got metric $metric for model $name trained with ${paramGrids(i)}.")
          metrics(i) = metric
          i += 1
        }
        log.info(s"Train validation split for $name metrics: {}", metrics.toSeq.mkString(","))
        val (bestMetric, bestIndex) =
          if (evaluator.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
          else metrics.zipWithIndex.minBy(_._1)
        log.info(s"Best set of parameters:\n${paramGrids(bestIndex)} for $name")
        log.info(s"Best train validation split metric: $bestMetric.")

        ValidatedModel(estimator, bestIndex, metrics, paramGrids)
    }
    trainingDataset.unpersist()
    validationDataset.unpersist()
    rdd.unpersist()

    val model =
      if (evaluator.isLargerBetter) groupedSummary.maxBy(_.bestMetric)
      else groupedSummary.minBy(_.bestMetric)

    val bestModel = model.model.fit(dataset, model.bestGrid).asInstanceOf[M]
    wrapBestModel(groupedSummary.toArray, bestModel, s"$trainRatio training split")
  }

  // TODO : Implement our own startified split method for better performance in a separate PR
  /**
   * Creates Train Validation Splits For TS
   * @param rdd
   * @return
   */
  private[op] override def createTrainValidationSplits(
    rdd: RDD[(Double, Vector, String)]): Array[(RDD[Row], RDD[Row])] = {

    val Array(trainData, validateData) = {
      if (stratify && isClassification) {
        log.info(s"Creating stratified train/validation with training ratio of $trainRatio")

        val classes = rdd.map(_._1).distinct().collect()
        // Creates RDD grouped by classes (0, 1, 2, 3, ..., K)
        val rddByClass = classes.map(label => rdd.filter(_._1 == label)
          .map { case (label, features, key) => key -> Seq(Row(label, features)) }.reduceByKey(_ ++ _))

        // Train/Validation data for each class
        val splitByClass = rddByClass.map(_.randomSplit(Array(trainRatio, 1 - trainRatio), seed)
          .map(_.values.flatMap(identity)))

        if (splitByClass.isEmpty) throw new Error("Train Validation Data Grouped by class is empty")
        // Merging Train/Validation data one by one
        splitByClass.reduce[Array[RDD[Row]]] {
          case (Array(train1: RDD[Row], validate1: RDD[Row]), Array(train2: RDD[Row], validate2: RDD[Row])) =>
            Array(train1.union(train2), validate1.union(validate2))
        }

      } else {
        rdd.map { case (label, features, key) => key -> Seq(Row(label, features)) }
          .reduceByKey(_ ++ _)
          .randomSplit(Array(trainRatio, 1 - trainRatio), seed)
          .map(_.values.flatMap(identity))
      }
    }
    Array((trainData, validateData))
  }
}
