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

package com.salesforce.op.utils.stages

import com.salesforce.op.features.OPFeature
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.{EstimatorType, ModelType}
import com.salesforce.op.stages.impl.selector.{HasTestEval, ModelSelector, ModelSelectorNames}
import com.salesforce.op.stages.{OPStage, OpTransformer}
import com.salesforce.op.{OpWorkflow, OpWorkflowModel}
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

/**
 * Functionality for manipulating stages DAG and fitting stages
 *
 * NOTE: this should be kept private to OP, cause we do not want users to mess up with
 * the internal mechanisms of our workflows.
 */
private[op] case object FitStagesUtil {

  /**
   * DAG layer - stages with their distance pairs
   */
  type Layer = Array[(OPStage, Int)]

  /**
   * Stages DAG - unique stages layered by distance (desc order)
   */
  type StagesDAG = Array[Layer]

  /**
   * Model Selector type
   */
  type MS = ModelSelector[_ <: ModelType, _ <: EstimatorType]

  /**
   * Fitted DAG together with it's trainding & test data
   *
   * @param trainData    train data
   * @param testData     test data
   * @param transformers fitted transformers
   */
  case class FittedDAG(trainData: Dataset[Row], testData: Dataset[Row], transformers: Array[OPStage])

  /**
   * Extracted Model Selector and Split the DAG into
   *
   * @param modelSelector maybe model selector (if any)
   * @param before        DAG before CV/TS
   * @param during        DAG during CV/TS
   * @param after         DAG after CV/TS
   */
  case class CutDAG(modelSelector: Option[(MS, Int)], before: StagesDAG, during: StagesDAG, after: StagesDAG)

  private val log = LoggerFactory.getLogger(this.getClass.getName.stripSuffix("$"))

  /**
   * Efficiently apply all op stages
   *
   * @param opStages list of op stages to apply
   * @param df       dataframe to apply them too
   * @return new data frame containing columns with output for all stages fed in
   */
  def applyOpTransformations(opStages: Array[_ <: OPStage with OpTransformer], df: Dataset[Row])
    (implicit spark: SparkSession): Dataset[Row] = {
    if (opStages.isEmpty) df
    else {
      log.info("Applying {} OP stage(s): {}", opStages.length, opStages.map(_.uid).mkString(","))

      val newSchema = opStages.foldLeft(df.schema) {
        case (schema, s) => s.setInputSchema(schema).transformSchema(schema)
      }
      val transforms = opStages.map(_.transformRow)
      val transformed: RDD[Row] =
        df.rdd.map { (row: Row) =>
          val values = new Array[Any](row.length + transforms.length)
          var i = 0
          while (i < values.length) {
            values(i) = if (i < row.length) row.get(i) else transforms(i - row.length)(row)
            i += 1
          }
          Row.fromSeq(values)
        }

      spark.createDataFrame(transformed, newSchema).persist()
    }
  }

  /**
   * Transform the data using the specified Spark transformers.
   * Applying all the transformers one by one as [[org.apache.spark.ml.Pipeline]] does.
   *
   * ATTENTION: This method applies transformers sequentially (as [[org.apache.spark.ml.Pipeline]] does)
   * and usually results in slower run times with large amount of transformations due to Catalyst crashes,
   * therefore always remember to set 'persistEveryKStages' to break up Catalyst.
   *
   * @param transformers        spark transformers to apply
   * @param persistEveryKStages how often to break up Catalyst by persisting the data,
   *                            to turn off set to Int.MaxValue (not recommended)
   * @return Dataframe transformed data
   */
  def applySparkTransformations(
    data: Dataset[Row], transformers: Array[Transformer], persistEveryKStages: Int
  )(implicit spark: SparkSession): Dataset[Row] = {

    // you have more than 5 stages and are not persisting at least once
    if (transformers.length > 5 && persistEveryKStages > transformers.length) {
      log.warn(
        "Number of transformers for scoring pipeline exceeds the persistence frequency. " +
          "Scoring performance may significantly degrade due to Catalyst optimizer issues. " +
          s"Consider setting 'persistEveryKStages' to a smaller number (ex. ${OpWorkflowModel.PersistEveryKStages}).")
    }

    // A holder for the last persisted rdd
    var lastPersisted: Option[RDD[_]] = None

    // Apply all the transformers one by one as [[org.apache.spark.ml.Pipeline]] does
    val transformedData: DataFrame =
      transformers.zipWithIndex.foldLeft(data) { case (df, (stage, i)) =>
        val persist = i > 0 && i % persistEveryKStages == 0
        log.info(s"Applying stage: ${stage.uid}{}", if (persist) " (persisted)" else "")
        val newDF = stage.asInstanceOf[Transformer].transform(df)
        if (!persist) newDF
        else {
          // Converting to rdd and back here to break up Catalyst [SPARK-13346]
          val persisted = newDF.rdd.persist()
          lastPersisted.foreach(_.unpersist())
          lastPersisted = Some(persisted)
          spark.createDataFrame(persisted, newDF.schema)
        }
      }
    transformedData
  }

  /**
   * Computes stages DAG
   *
   * @param features array if features in workflow
   * @return unique stages layered by distance (desc order)
   */
  def computeDAG(features: Array[OPFeature]): StagesDAG = {
    val (failures, parents) = features.map(_.parentStages()).partition(_.isFailure)

    if (failures.nonEmpty) {
      throw new IllegalArgumentException("Failed to compute stages DAG", failures.head.failed.get)
    }

    // Stages sorted by distance
    val sortedByDistance: Array[(OPStage, Int)] = parents.flatMap(_.get)

    // Stages layered by distance
    val layeredByDistance: StagesDAG =
      sortedByDistance.groupBy(_._2).toArray
        .map(_._2.sortBy(_._1.getOutputFeatureName))
        .sortBy(s => -s.head._2)

    // Unique stages layered by distance
    layeredByDistance
      .foldLeft(Set.empty[OPStage], Array.empty[Array[(OPStage, Int)]]) {
        case ((seen, filtered), uncleaned) =>
          // filter out any seen stages. also add distinct to filter out any duplicate stages in layer
          val unseen = uncleaned.filterNot(v => seen.contains(v._1)).distinct
          val nowSeen = seen ++ unseen.map(_._1)
          (nowSeen, filtered :+ unseen)
      }._2
  }

  /**
   * Fit DAG and apply transformations on data up to the last estimator stage
   *
   * @param dag                  DAG to fit
   * @param train                training dataset
   * @param test                 test dataset
   * @param hasTest              whether the test dataset is empty or not
   * @param indexOfLastEstimator Optional index of the last estimator
   * @param persistEveryKStages  frequency of persisting stages
   * @param fittedTransformers   list of already fitted transformers
   * @param spark                Spark session
   * @return Fitted and Transformed train/test before the last estimator with fitted transformers
   */
  def fitAndTransformDAG(
    dag: StagesDAG,
    train: Dataset[Row],
    test: Dataset[Row],
    hasTest: Boolean,
    indexOfLastEstimator: Option[Int],
    persistEveryKStages: Int = OpWorkflowModel.PersistEveryKStages,
    fittedTransformers: Seq[OPStage] = Seq.empty
  )(implicit spark: SparkSession): FittedDAG = {
    val alreadyFitted: ListBuffer[OPStage] = ListBuffer(fittedTransformers: _*)
    val (newTrain, newTest) =
      dag.foldLeft(train -> test) { case ((currTrain, currTest), stagesLayer) =>
        val index = stagesLayer.head._2
        val FittedDAG(newTrain, newTest, justFitted) = fitAndTransformLayer(
          stagesLayer = stagesLayer,
          train = currTrain,
          test = currTest,
          hasTest = hasTest,
          transformData = indexOfLastEstimator.exists(_ < index), // only need to update for fit before last estimator
          persistEveryKStages = persistEveryKStages
        )
        alreadyFitted ++= justFitted
        newTrain -> newTest
      }

    FittedDAG(newTrain, newTest, alreadyFitted.toArray)
  }

  /**
   * Fit a sequence of stages and transform a training and test dataset for use this function assumes all
   * stages fed in are on the same level of the dag
   *
   * @param train               training dataset for estimators
   * @param test                test dataset for evaluation
   * @param hasTest             whether the test dataset is empty or not
   * @param stagesLayer         stages to fit
   * @param transformData       should the input data be transformed or only used for fitting
   * @param persistEveryKStages persist data at this frequency during transformations
   * @return dataframes for train and test as well as the fitted stages
   */
  private def fitAndTransformLayer(
    stagesLayer: Layer,
    train: Dataset[Row],
    test: Dataset[Row],
    hasTest: Boolean,
    transformData: Boolean,
    persistEveryKStages: Int
  )(implicit spark: SparkSession): FittedDAG = {
    val stages = stagesLayer.map(_._1)
    val (estimators, noFit) = stages.partition(_.isInstanceOf[Estimator[_]])
    val fitEstimators = estimators.map { case e: Estimator[_] =>
      e.fit(train) match {
        case m: HasTestEval if hasTest =>
          m.evaluateModel(test)
          m.asInstanceOf[OPStage]
        case m =>
          m.asInstanceOf[OPStage]
      }
    }
    val transformers = noFit ++ fitEstimators

    val opTransformers = transformers.collect { case s: OPStage with OpTransformer => s }
    val sparkTransformers = transformers.collect {
      case s: Transformer if !s.isInstanceOf[OpTransformer] => s.asInstanceOf[Transformer]
    }

    if (transformData) {
      val withOPTrain = applyOpTransformations(opTransformers, train)
      val withAllTrain = applySparkTransformations(withOPTrain, sparkTransformers, persistEveryKStages)

      val withAllTest = if (hasTest) {
        val withOPTest = applyOpTransformations(opTransformers, test)
        applySparkTransformations(withOPTest, sparkTransformers, persistEveryKStages)
      } else test

      FittedDAG(trainData = withAllTrain, testData = withAllTest, transformers = transformers)
    } else {
      FittedDAG(trainData = train, testData = test, transformers = transformers)
    }
  }

  /**
   * Method that cut DAG in order to perform proper CV/TS.
   * Extracts Model Selector and Split the DAG into
   * 1. DAG before CV/TS
   * 2. DAG during CV/TS
   * 3. DAG after CV/TS
   *
   * @param dag DAG in the workflow to be cut
   * @return (Model Selector, nonCVTS DAG -to be done outside of CV/TS, CVTS DAG -to apply in the CV/TS)
   */
  def cutDAG(dag: StagesDAG): CutDAG = {
    if (dag.isEmpty) CutDAG(None, Array(), Array(), Array())
    else {
      // creates Array containing every Model Selector in the DAG
      val modelSelectorArrays = dag.flatten.collect { case (ms: MS, dist: Int) => (ms, dist) }
      val modelSelector = modelSelectorArrays.toList match {
        case Nil => None
        case List(ms) => Option(ms)
        case modelSelectors => throw new IllegalArgumentException(
          s"OpWorkflow can contain at most 1 Model Selector. Found ${modelSelectors.length} Model Selectors :" +
            s" ${modelSelectors.map(_._1).mkString(",")}")
      }

      // nonCVTS and CVTS DAGs
      val (nonCVTSDAG: StagesDAG, inCVTSDAG: StagesDAG, afterCVTSDAG: StagesDAG) =
        modelSelector.map { case (ms, dist) =>
          // Optimize the DAG by removing stages unrelated to ModelSelector

          // Create the DAG after Model Selector.
          val (afterCVTSDAG, beforeCVDAG) = dag.partition(_.exists(_._2 < dist))

          val modelSelectorDAG = computeDAG(Array(ms.getOutput()))
            .dropRight(1)
            .map(_.map{ case (stage, dist) => (stage, dist + afterCVTSDAG.length) })

          // Create the DAG without Model Selector. It will be used to compute the final nonCVTS DAG.
          val nonMSDAG: StagesDAG = beforeCVDAG.map(_.filterNot(_._1.isInstanceOf[MS])).filter(_.nonEmpty)

          // Index of first CVTS stage in ModelSelector DAG
          val firstCVTSIndex = modelSelectorDAG.indexWhere(_.exists(stage => {
            val inputs = stage._1.getTransientFeatures()
            inputs.exists(_.isResponse) && inputs.exists(!_.isResponse)
          }))

          // If no CVTS stages, the whole DAG is not in the CV/TS
          if (firstCVTSIndex == -1) (nonMSDAG, Array.empty[Layer], afterCVTSDAG) else {

            val cVTSDAG = modelSelectorDAG.drop(firstCVTSIndex)

            // nonCVTSDAG is the complementary DAG
            // The rule is "nonCVTSDAG = nonMSDAG - CVTSDAG"
            val nonCVTSDAG = {
              val flattenedCVTSDAG = cVTSDAG.flatten.map(_._1)
              nonMSDAG.map(_.filterNot { case (stage: OPStage, _) => flattenedCVTSDAG.contains(stage) })
                .filter(_.nonEmpty) // Remove empty layers
            }

            (nonCVTSDAG, cVTSDAG, afterCVTSDAG)
          }
        }.getOrElse((Array.empty[Layer], Array.empty[Layer], Array.empty[Layer]))

      CutDAG(modelSelector, before = nonCVTSDAG, during = inCVTSDAG, after = afterCVTSDAG)
    }
  }

  /**
   * Method that cut DAG in order to perform proper CV/TS.
   * Extracts Model Selector and Split the DAG into
   * 1. DAG before CV/TS
   * 2. DAG during CV/TS
   * 3. DAG after CV/TS
   *
   * @param wf to be cut
   * @return (Model Selector, nonCVTS DAG -to be done outside of CV/TS, CVTS DAG -to apply in the CV/TS)
   */
  def cutDAG(wf: OpWorkflow): CutDAG = cutDAG(computeDAG(wf.getResultFeatures()))

}
