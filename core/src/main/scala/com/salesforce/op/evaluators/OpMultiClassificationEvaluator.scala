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

package com.salesforce.op.evaluators

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Dataset
import org.slf4j.LoggerFactory
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

/**
 *
 * Instance to evaluate Multi Classification metrics
 * The metrics are  Precision, Recall, F1 and Error Rate
 * Default evaluation returns F1 score
 *
 * @param name name of default metric
 * @param isLargerBetter is metric better if larger
 * @param uid uid for instance
 */
private[op] class OpMultiClassificationEvaluator
(
  override val name: String = OpEvaluatorNames.multi,
  override val isLargerBetter: Boolean = true,
  override val uid: String = UID[OpMultiClassificationEvaluator]
) extends OpMultiClassificationEvaluatorBase[MultiClassificationMetrics](uid) {

  private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: MultiClassificationMetrics => Double = _.F1

  override def evaluateAll(data: Dataset[_]): MultiClassificationMetrics = {
    val (labelColName, predictionColName, rawPredictionColName) = (getLabelCol, getPredictionCol, getRawPredictionCol)

    log.debug(
      "Evaluating metrics on columns :\n label : {}\n rawPrediction : {}\n prediction : {}\n",
      labelColName, rawPredictionColName, predictionColName
    )

    import data.sparkSession.implicits._
    val rdd = data.select(predictionColName, labelColName).as[(Double, Double)].rdd

    val multiclassMetrics = new MulticlassMetrics(rdd)
    val error = 1.0 - multiclassMetrics.accuracy
    val precision = multiclassMetrics.weightedPrecision
    val recall = multiclassMetrics.weightedRecall
    val f1 = if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)

    val metrics = MultiClassificationMetrics(Precision = precision, Recall = recall, F1 = f1, Error = error)

    log.info("Evaluated metrics: {}", metrics.toString)
    metrics
  }


  final private[op] def getMultiEvaluatorMetric(metricName: ClassificationEvalMetric, dataset: Dataset[_]): Double = {
    new MulticlassClassificationEvaluator()
      .setLabelCol(getLabelCol)
      .setPredictionCol(getPredictionCol)
      .setMetricName(metricName.sparkEntryName)
      .evaluate(dataset)
  }

}


/**
 * Metrics of MultiClassification Problem
 *
 * @param Precision
 * @param Recall
 * @param F1
 * @param Error
 */
case class MultiClassificationMetrics(Precision: Double, Recall: Double, F1: Double, Error: Double)
  extends EvaluationMetrics
