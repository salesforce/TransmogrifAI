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

package com.salesforce.op.utils.stages

import com.salesforce.op.OpWorkflowModel
import com.salesforce.op.stages.{OPStage, OpTransformer}
import com.salesforce.op.stages.impl.selector.HasTestEval
import org.apache.spark.ml.{Estimator, Transformer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.Logger
import com.salesforce.op.utils.spark.RichDataset._

private[op] case object FitStagesUtil {

  /**
   * Efficiently apply all op stages
   * @param opStages  list of op stages to apply
   * @param df dataframe to apply them too
   * @return new data frame containing columns with output for all stages fed in
   */
  def applyOpTransformations(opStages: Array[_ <:OPStage with OpTransformer], df: DataFrame)
    (implicit spark: SparkSession, log: Logger): DataFrame = {
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
    data: DataFrame, transformers: Array[Transformer], persistEveryKStages: Int
  )(implicit spark: SparkSession, log: Logger): DataFrame = {

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
   * Fit a sequence of stages and transform a training and test dataset for use this function assumes all
   * stages fed in are on the same level of the dag
   * @param train training dataset for estimators
   * @param test test dataset for evaluation
   * @param stages stages to fix
   * @param transformData should the imput data be transformed or only used for fitting
   * @param persistEveryKStages persist data at this frequency during transformations
   * @param doTest test data is nonempty
   * @return dataframes for train and test as well as the fitted stages
   */
  def fitAndTransform(
    train: DataFrame,
    test: DataFrame,
    stages: Array[(OPStage)],
    transformData: Boolean,
    persistEveryKStages: Int,
    doTest: Option[Boolean] = None
  )(implicit spark: SparkSession, log: Logger): (DataFrame, DataFrame, Array[OPStage]) = {

    val testExists = doTest.getOrElse(!test.isEmpty)
    val (estimators, noFit) = stages.partition( _.isInstanceOf[Estimator[_]] )
    val fitEstimators = estimators.map { case e: Estimator[_] =>
      e.fit(train) match {
        case m: HasTestEval if testExists => m.evaluateModel(test)
          m.asInstanceOf[OPStage]
        case m => m.asInstanceOf[OPStage]
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

      val withAllTest = if (testExists) {
        val withOPTest = applyOpTransformations(opTransformers, test)
        applySparkTransformations(withOPTest, sparkTransformers, persistEveryKStages)
      } else test

      (withAllTrain, withAllTest, transformers)
    } else {
      (train, test, transformers)
    }
  }


}
