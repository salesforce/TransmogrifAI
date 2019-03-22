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

package com.salesforce.op.evaluators

import com.fasterxml.jackson.databind.annotation.JsonDeserialize
import com.salesforce.op.UID
import com.salesforce.op.utils.spark.RichEvaluator._
import com.salesforce.op.evaluators.BinaryClassEvalMetrics._
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics => SparkMLBinaryClassificationMetrics}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}
import org.slf4j.LoggerFactory

/**
 *
 * Instance to evaluate Binary Classification metrics
 * The metrics are AUROC, AUPR, Precision, Recall, F1 and Error Rate
 * Default evaluation returns AUROC
 *
 * @param name           name of default metric
 * @param isLargerBetter is metric better if larger
 * @param uid            uid for instance
 */

private[op] class OpBinaryClassificationEvaluator
(
  override val name: EvalMetric = OpEvaluatorNames.Binary,
  override val isLargerBetter: Boolean = true,
  override val uid: String = UID[OpBinaryClassificationEvaluator],
  val numBins: Int = 100
) extends OpBinaryClassificationEvaluatorBase[BinaryClassificationMetrics](uid = uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: BinaryClassificationMetrics => Double = _.AuROC

  override def evaluateAll(data: Dataset[_]): BinaryClassificationMetrics = {
    val labelColName = getLabelCol
    val dataUse = makeDataToUse(data, labelColName)

    val (rawPredictionColName, predictionColName, probabilityColName) =
      (getRawPredictionCol, getPredictionValueCol, getProbabilityCol)

    log.debug(
      "Evaluating metrics on columns :\n label : {}\n rawPrediction : {}\n prediction : {}\n probability : {}\n",
      labelColName, rawPredictionColName, predictionColName, probabilityColName
    )

    import dataUse.sparkSession.implicits._
    val rdd = dataUse.select(predictionColName, labelColName).as[(Double, Double)].rdd

    if (rdd.isEmpty()) {
      log.warn("The dataset is empty. Returning empty metrics.")
      BinaryClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        Seq(), Seq(), Seq(), Seq(), LiftMetrics.empty)
    } else {
      val multiclassMetrics = new MulticlassMetrics(rdd)
      val labels = multiclassMetrics.labels

      val (tn, fn, fp, tp) = if (labels.length == 2) {
        val confusionMatrix = multiclassMetrics.confusionMatrix
        (confusionMatrix(0, 0), confusionMatrix(1, 0), confusionMatrix(0, 1), confusionMatrix(1, 1))
      } else {
        // if labels and predictions of data are only one class, cannot compute confusion matrix
        val size = dataUse.count().toDouble
        if (labels.head == 0.0) (size, 0.0, 0.0, 0.0) else (0.0, 0.0, 0.0, size)
      }

      val precision = if (tp + fp == 0.0) 0.0 else tp / (tp + fp)
      val recall = if (tp + fn == 0.0) 0.0 else tp / (tp + fn)
      val f1 = if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)
      val error = if (tp + fp + tn + fn == 0.0) 0.0 else (fp + fn) / (tp + fp + tn + fn)

      val scoreAndLabels =
        dataUse.select(col(probabilityColName), col(labelColName).cast(DoubleType)).rdd.map {
          case Row(prob: Vector, label: Double) => (prob(1), label)
          case Row(prob: Double, label: Double) => (prob, label)
        }
      val sparkMLMetrics = new SparkMLBinaryClassificationMetrics(scoreAndLabels = scoreAndLabels, numBins = numBins)
      val thresholds = sparkMLMetrics.thresholds().collect()
      val precisionByThreshold = sparkMLMetrics.precisionByThreshold().collect().map(_._2)
      val recallByThreshold = sparkMLMetrics.recallByThreshold().collect().map(_._2)
      val falsePositiveRateByThreshold = sparkMLMetrics.roc().collect().map(_._1).slice(1, thresholds.length + 1)
      val aUROC = sparkMLMetrics.areaUnderROC()
      val aUPR = sparkMLMetrics.areaUnderPR()
      val liftMetrics = new LiftEvaluator().evaluateScoreAndLabels(scoreAndLabels = scoreAndLabels)
      val metrics = BinaryClassificationMetrics(
        Precision = precision, Recall = recall, F1 = f1, AuROC = aUROC,
        AuPR = aUPR, Error = error, TP = tp, TN = tn, FP = fp, FN = fn,
        thresholds, precisionByThreshold, recallByThreshold, falsePositiveRateByThreshold,
        LiftMetrics = liftMetrics
      )
      log.info("Evaluated metrics: {}", metrics.toString)
      metrics
    }
  }

  final protected def getBinaryEvaluatorMetric(
                                                metricName: ClassificationEvalMetric,
                                                dataset: Dataset[_],
                                                default: => Double
                                              ): Double = {
    import dataset.sparkSession.implicits._
    val labelColName = getLabelCol
    val dataUse = makeDataToUse(dataset, labelColName)
    lazy val rdd = dataUse.select(getPredictionValueCol, labelColName).as[(Double, Double)].rdd
    lazy val noData = rdd.isEmpty()

    metricName match {
      case AuPR | AuROC =>
        new BinaryClassificationEvaluator()
          .setLabelCol(labelColName)
          .setRawPredictionCol(getRawPredictionCol)
          .setMetricName(metricName.sparkEntryName)
          .evaluateOrDefault(dataUse, default = default)

      case Error =>
        new MulticlassClassificationEvaluator()
          .setLabelCol(labelColName)
          .setPredictionCol(getPredictionValueCol)
          .setMetricName(metricName.sparkEntryName)
          .evaluateOrDefault(dataUse, default = default)

      case Precision | Recall | F1 if noData => default
      case Precision => new MulticlassMetrics(rdd).precision(1.0)
      case Recall => new MulticlassMetrics(rdd).recall(1.0)
      case F1 => new MulticlassMetrics(rdd).fMeasure(1.0)

      case m =>
        throw new IllegalArgumentException(s"Unsupported binary evaluation metric $m")
    }
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
  FN: Double,
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  thresholds: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  precisionByThreshold: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  recallByThreshold: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  falsePositiveRateByThreshold: Seq[Double],
  LiftMetrics: LiftMetrics
) extends EvaluationMetrics {
  def rocCurve: Seq[(Double, Double)] = recallByThreshold.zip(falsePositiveRateByThreshold)

  def prCurve: Seq[(Double, Double)] = precisionByThreshold.zip(recallByThreshold)
}
