/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.evaluators

import com.salesforce.op.UID
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Dataset
import org.slf4j.LoggerFactory

/**
 *
 * Instance to evaluate Binary Classification metrics
 * The metrics are AUROC, AUPR, Precision, Recall, F1 and Error Rate
 * Default evaluation returns AUROC
 *
 * @param name name of default metric
 * @param isLargerBetter is metric better if larger
 * @param uid uid for instance
 */

private[op] class OpBinaryClassificationEvaluator
(
  override val name: String = OpEvaluatorNames.binary,
  override val isLargerBetter: Boolean = true,
  override val uid: String = UID[OpBinaryClassificationEvaluator]
) extends OpBinaryClassificationEvaluatorBase[BinaryClassificationMetrics](uid = uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: BinaryClassificationMetrics => Double = _.AuROC

  override def evaluateAll(data: Dataset[_]): BinaryClassificationMetrics = {
    val (labelColName, rawPredictionColName, predictionColName) = (getLabelCol, getRawPredictionCol, getPredictionCol)

    log.debug(
      "Evaluating metrics on columns :\n label : {}\n rawPrediction : {}\n prediction : {}\n",
      labelColName, rawPredictionColName, predictionColName
    )

    val Array(aUROC, aUPR) = Array(
      BinaryClassEvalMetrics.AuROC,
      BinaryClassEvalMetrics.AuPR
    ).map(getBinaryEvaluatorMetric(_, data))


    // scalastyle:off
    import data.sparkSession.implicits._
    // scalastyle:on

    val rdd = data.select(predictionColName, labelColName).as[(Double, Double)].rdd

  if (rdd.isEmpty()) {
    log.error("The dataset is empty")
    BinaryClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  } else {
    val multiclassMetrics = new MulticlassMetrics(rdd)

    val labels = multiclassMetrics.labels

    val (tn, fn, fp, tp) = if (labels.length == 2) {

      val confusionMatrix = multiclassMetrics.confusionMatrix
      (confusionMatrix(0, 0), confusionMatrix(1, 0), confusionMatrix(0, 1), confusionMatrix(1, 1))
    } else {
      // if labels and predictions of data are only one class, cannot compute confusion matrix
      val size = data.count().toDouble
      if (labels.head == 0.0) {
        (size, 0.0, 0.0, 0.0)
      } else (0.0, 0.0, 0.0, size)
    }

    val precision = if (tp + fp == 0.0) 0.0 else tp / (tp + fp)
    val recall = if (tp + fn == 0.0) 0.0 else tp / (tp + fn)
    val f1 = if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)
    val error = if (tp + fp + tn + fn == 0.0) 0.0 else (fp + fn) / (tp + fp + tn + fn)

    val metrics = BinaryClassificationMetrics(
      Precision = precision,
      Recall = recall,
      F1 = f1,
      AuROC = aUROC,
      AuPR = aUPR,
      Error = error,
      TP = tp,
      TN = tn,
      FP = fp,
      FN = fn
    )

    log.info("Evaluated metrics : {}", metrics.toString)
    metrics
  }
  }

  final protected def getBinaryEvaluatorMetric(metricName: ClassificationEvalMetric, dataset: Dataset[_]): Double = {
    new BinaryClassificationEvaluator()
      .setLabelCol(getLabelCol)
      .setRawPredictionCol(getRawPredictionCol)
      .setMetricName(metricName.sparkEntryName)
      .evaluate(dataset)
  }

  final protected def getMultiEvaluatorMetric(metricName: ClassificationEvalMetric, dataset: Dataset[_]): Double = {
    new MulticlassClassificationEvaluator()
      .setLabelCol(getLabelCol)
      .setPredictionCol(getPredictionCol)
      .setMetricName(metricName.sparkEntryName)
      .evaluate(dataset)
  }
}


/**
 * Metrics of Binary Classification Problem
 *
 * @param Precision
 * @param Recall
 * @param F1
 * @param AuROC
 * @param AuPR
 * @param Error
 * @param TP
 * @param TN
 * @param FP
 * @param FN
 */
case class BinaryClassificationMetrics
(
  Precision: Double,
  Recall: Double,
  F1: Double,
  AuROC: Double,
  AuPR: Double,
  Error: Double,
  TP: Double,
  TN: Double,
  FP: Double,
  FN: Double
) extends EvaluationMetrics
