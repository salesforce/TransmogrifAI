/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
