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
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.StructType
import com.salesforce.op.stages.impl.selector.{ModelInfo, ModelSelectorBaseNames, StageParamNames}
import com.salesforce.op.stages.impl.tuning.SelectorData.LabelFeaturesKey
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Dataset, Row}
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.Vector

import scala.collection.parallel.mutable.ParArray


private[impl] class OpCrossValidation[M <: Model[_], E <: Estimator[_]]
(
  val numFolds: Int = ValidatorParamDefaults.NumFolds,
  val seed: Long = ValidatorParamDefaults.Seed,
  val evaluator: OpEvaluatorBase[_],
  val stratify: Boolean = ValidatorParamDefaults.Stratify
) extends OpValidator[M, E] {

  val validationName: String = ModelSelectorBaseNames.CrossValResults
  private val blas = BLAS.getInstance()

  private def findBestModel(
    folds: ParArray[(E, Array[Double], Array[ParamMap])]
  ): ValidatedModel[E] = {
    val metrics = folds.map(_._2).reduce(_ + _)
    blas.dscal(metrics.length, 1.0 / numFolds, metrics, 1)
    val (est, _, grid) = folds.head
    log.info(s"Average cross-validation for $est metrics: {}", metrics.toSeq.mkString(","))
    val (bestMetric, bestIndex) =
      if (evaluator.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    log.info(s"Best set of parameters:\n${grid(bestIndex)}")
    log.info(s"Best cross-validation metric: $bestMetric.")
    ValidatedModel(est, bestIndex, metrics, grid)
  }

  // TODO use futures to parallelize https://github.com/apache/spark/commit/16c4c03c71394ab30c8edaf4418973e1a2c5ebfe
  private[op] def validate(
    modelInfo: Seq[ModelInfo[E]],
    dataset: Dataset[_],
    label: String,
    features: String
  ): BestModel[M] = {

    // get param that stores the label column
    val labelCol = evaluator.getParam(ValidatorParamDefaults.labelCol)
    evaluator.set(labelCol, label)

    val sparkSession = dataset.sparkSession
    import sparkSession.implicits._
    val rdd = dataset.as[LabelFeaturesKey].rdd.persist()


    // creating k train/validation data
    val splits: Array[(RDD[Row], RDD[Row])] = createTrainValidationSplits(rdd)


    val schema = dataset.schema
    val newSchema = StructType(schema.dropRight(1)) // dropping key

    val modelWithGrid = modelInfo.map(m => (m.sparkEstimator, m.grid.build(), m.modelName))

    val fitSummary = splits.zipWithIndex.par.flatMap {
      case ((training, validation), splitIndex) =>

        log.info(s"Cross Validation $splitIndex with multiple sets of parameters.")
        val trainingDataset = sparkSession.createDataFrame(training, newSchema).persist()
        val validationDataset = sparkSession.createDataFrame(validation, newSchema).persist()

        val summary = modelWithGrid.map {
          case (estimator, paramGrids, name) =>
            val pi1 = estimator.getParam(StageParamNames.inputParam1Name)
            val pi2 = estimator.getParam(StageParamNames.inputParam2Name)
            estimator.set(pi1, label).set(pi2, features)

            val numModels = paramGrids.length
            val metrics = new Array[Double](paramGrids.length)

            // multi-model training
            val models = estimator.fit(trainingDataset, paramGrids).asInstanceOf[Seq[M]]
            var i = 0
            while (i < numModels) {
              val metric = evaluator.evaluate(models(i).transform(validationDataset, paramGrids(i)))
              log.debug(s"Got metric $metric for $name trained with ${paramGrids(i)}.")
              metrics(i) = metric
              i += 1
            }
            (estimator, metrics, paramGrids)
        }
        trainingDataset.unpersist()
        validationDataset.unpersist()
        summary
    }
    rdd.unpersist()

    val groupedSummary = fitSummary.groupBy(_._1).map { case (_, folds) => findBestModel(folds) }.toArray

    val model =
      if (evaluator.isLargerBetter) groupedSummary.maxBy(_.bestMetric)
      else groupedSummary.minBy(_.bestMetric)

    val bestModel = model.model.fit(dataset, model.bestGrid).asInstanceOf[M]
    wrapBestModel(groupedSummary, bestModel, s"$numFolds folds")
  }

  // TODO : Implement our own kFold method for better performance in a separate PR
  /**
   * Creates Train Validation Splits For CV
   * @param rdd
   * @return Array((TrainRDD, ValidationRDD), Index)
   */
  private[op] override def createTrainValidationSplits(
    rdd: RDD[(Double, Vector, String)]): Array[(RDD[Row], RDD[Row])] = {

    if (stratify && isClassification) {
      log.info(s"Creating $numFolds stratified folds")
      val classes = rdd.map(_._1).distinct().collect()
      // Creates RDD grouped by classes (0, 1, 2, 3, ..., K)
      val rddByClass = classes.map(label => rdd.filter(_._1 == label)
        .map { case (label, features, key) => key -> Seq(Row(label, features)) }.reduceByKey(_ ++ _))

      // Cross Validation's Train/Validation data for each class
      val foldsByClass = rddByClass.map { case rdd: RDD[(String, Seq[Row])] => {
        MLUtils.kFold(rdd, numFolds, seed)
          .map { case (rdd1, rdd2) => (rdd1.values.flatMap(identity), rdd2.values.flatMap(identity)) }
      }
      }.toSeq

      if (foldsByClass.isEmpty) throw new Error("Train Validation Data Grouped by class is empty")
      // Merging Train/Validation data one by one
      foldsByClass.reduce[Array[(RDD[Row], RDD[Row])]] {
        // cv1 and cv2 are arrays of train/validation data
        case (cv1: Array[(RDD[Row], RDD[Row])], cv2: Array[(RDD[Row], RDD[Row])]) =>
          (cv1 zip cv2).map { // zip the two arrays and merge the tuples one by one
            case ((train1: RDD[Row], test1: RDD[Row]), (train2: RDD[Row], test2: RDD[Row])) =>
              (train1.union(train2), test1.union(test2))
          }
      }
    } else {
      val rddRow = rdd.map { case (label, features, key) => key -> Seq(Row(label, features)) }.reduceByKey(_ ++ _)
      MLUtils.kFold(rddRow, numFolds, seed)
        .map { case (rdd1, rdd2) => (rdd1.values.flatMap(identity), rdd2.values.flatMap(identity)) }
    }
  }

}
