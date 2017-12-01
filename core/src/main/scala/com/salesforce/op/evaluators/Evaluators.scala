/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.evaluators

import com.salesforce.op.UID
import com.salesforce.op.features.types.OPVector
import com.salesforce.op.utils.json.JsonUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Dataset

/**
 * Just a handy factory for evaluators
 */
object Evaluators {

  /**
   * Factory that performs the evaluation of metrics for Binary Classification
   * The metrics returned are AUROC, AUPR, Precision, Recall, F1 and Error Rate
   */
  object BinaryClassification {

    /**
     * default: Area under ROC
     */
    def apply(): OpBinaryClassificationEvaluator = auROC()

    /**
     * Area under ROC
     */
    def auROC(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = OpMetricsNames.auROC, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getBinaryEvaluatorMetric(BinaryClassEvalMetrics.AuROC, dataset)
      }

    /**
     * Area under Precision/Recall curve
     */
    def auPR(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = OpMetricsNames.auPR, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getBinaryEvaluatorMetric(BinaryClassEvalMetrics.AuPR, dataset)
      }

    /**
     * Precision
     */
    def precision(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = OpMetricsNames.precision, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double = {
          // scalastyle:off
          import dataset.sparkSession.implicits._
          // scalastyle:on
          new MulticlassMetrics(dataset.select(getPredictionCol, getLabelCol).as[(Double, Double)].rdd).precision(1.0)
        }
      }

    /**
     * Recall
     */
    def recall(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = OpMetricsNames.recall, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double = {
          // scalastyle:off
          import dataset.sparkSession.implicits._
          // scalastyle:on
          new MulticlassMetrics(dataset.select(getPredictionCol, getLabelCol).as[(Double, Double)].rdd).recall(1.0)
        }
      }

    /**
     * F1 score
     */
    def f1(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = OpMetricsNames.f1, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double = {
          // scalastyle:off
          import dataset.sparkSession.implicits._
          // scalastyle:on
          new MulticlassMetrics(
            dataset.select(getPredictionCol, getLabelCol).as[(Double, Double)].rdd).fMeasure(1.0)
        }
      }

    /**
     * Prediction error
     */
    def error(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = OpMetricsNames.error, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          1.0 - getMultiEvaluatorMetric(MultiClassEvalMetrics.Error, dataset)
      }

    /**
     * Custom binary evaluator
     *
     * @param metricName     name of default metric
     * @param isLargerBetter is the default metric better when larger or smaller
     * @param evaluateFn     evaluate function that returns one metric.
     *                       Note: dataset consists of four columns: (label, raw prediction, probability, prediction)
     * @return a binary evaluator
     */
    def custom(
      metricName: String,
      isLargerBetter: Boolean = true,
      evaluateFn: Dataset[(Double, OPVector#Value, OPVector#Value, Double)] => Double
    ): OpBinaryClassificationEvaluatorBase[SingleMetric] = {
      val islbt = isLargerBetter
      new OpBinaryClassificationEvaluatorBase[SingleMetric](
        uid = UID[OpBinaryClassificationEvaluatorBase[SingleMetric]]
      ) {
        override val name: String = metricName
        override val isLargerBetter: Boolean = islbt

        override def getDefaultMetric: SingleMetric => Double = _.value

        override def evaluateAll(dataset: Dataset[_]): SingleMetric = {
          // scalastyle:off
          import dataset.sparkSession.implicits._
          // scalastyle:on
          val ds = dataset.select(getLabelCol, getRawPredictionCol, getProbabilityCol, getPredictionCol)
            .as[(Double, OPVector#Value, OPVector#Value, Double)]
          val metric = evaluateFn(ds)
          SingleMetric(name, metric)
        }
      }
    }
  }


  /**
   * Factory that performs the evaluation of metrics for Binary Classification
   * The metrics returned are Precision, Recall, F1 and Error Rate
   */
  object MultiClassification {

    /**
     * default: F1 Score
     */
    def apply(): OpMultiClassificationEvaluator = f1()

    /**
     * Weighted Precision
     */
    def precision(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = OpMetricsNames.precision, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getMultiEvaluatorMetric(MultiClassEvalMetrics.Precision, dataset)
      }

    /**
     * Weighted Recall
     */
    def recall(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = OpMetricsNames.recall, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getMultiEvaluatorMetric(MultiClassEvalMetrics.Recall, dataset)
      }

    /**
     * F1 Score
     */
    def f1(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = OpMetricsNames.f1, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getMultiEvaluatorMetric(MultiClassEvalMetrics.F1, dataset)
      }

    /**
     * Prediction Error
     */
    def error(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = OpMetricsNames.error, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          1.0 - getMultiEvaluatorMetric(MultiClassEvalMetrics.Error, dataset)
      }

    /**
     * Custom multiclass evaluator
     *
     * @param metricName     name of default metric
     * @param isLargerBetter is the default metric better when larger or smaller
     * @param evaluateFn     evaluate function that returns one metric enclosed in a case class.
     *                       Note: dataset consists of four columns: (label, raw prediction, probability, prediction)
     * @return a new multiclass evaluator
     */
    def custom(
      metricName: String,
      isLargerBetter: Boolean = true,
      evaluateFn: Dataset[(Double, OPVector#Value, OPVector#Value, Double)] => Double
    ): OpMultiClassificationEvaluatorBase[SingleMetric] = {
      val islbt = isLargerBetter
      new OpMultiClassificationEvaluatorBase[SingleMetric](
        uid = UID[OpMultiClassificationEvaluatorBase[SingleMetric]]
      ) {
        override val name: String = metricName
        override val isLargerBetter: Boolean = islbt

        override def getDefaultMetric: SingleMetric => Double = _.value

        override def evaluateAll(dataset: Dataset[_]): SingleMetric = {
          // scalastyle:off
          import dataset.sparkSession.implicits._
          // scalastyle:on
          val ds = dataset.select(getLabelCol, getRawPredictionCol, getProbabilityCol, getPredictionCol)
            .as[(Double, OPVector#Value, OPVector#Value, Double)]
          try {
            val metric = evaluateFn(ds)
            SingleMetric(name, metric)
          } catch {
            case iae: IllegalArgumentException =>
              val size = dataset.count
              val desc = s"dataset with ($getLabelCol, $getRawPredictionCol, $getProbabilityCol, $getPredictionCol)"
              val msg = if (size == 0) {
                s"empty $desc"
              } else {
                s"$desc of $size rows"
              }
              throw new IllegalArgumentException(s"Metric $name failed on $msg", iae)
          }
        }
      }
    }

  }

  /**
   * Factory that performs the evaluation of metrics for Regression
   * The metrics are rmse, mse, r2 and mae
   */
  object Regression {

    /**
     * default: Root Mean Squared Error
     */
    def apply(): OpRegressionEvaluator = rmse()

    /**
     * Mean Squared Error
     */
    def mse(): OpRegressionEvaluator =
      new OpRegressionEvaluator(name = OpMetricsNames.meanSquaredError, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.MeanSquaredError, dataset)
      }

    /**
     * Mean Absolute Error
     */
    def mae(): OpRegressionEvaluator =
      new OpRegressionEvaluator(name = OpMetricsNames.meanAbsoluteError, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.MeanAbsoluteError, dataset)
      }

    /**
     * R2
     */
    def r2(): OpRegressionEvaluator =
      new OpRegressionEvaluator(name = OpMetricsNames.r2, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.R2, dataset)
      }

    /**
     * Root Mean Squared Error
     */
    def rmse(): OpRegressionEvaluator =
      new OpRegressionEvaluator(name = OpMetricsNames.rootMeanSquaredError, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.RootMeanSquaredError, dataset)
      }

    /**
     * Custom regression evaluator
     *
     * @param metricName     name of default metric
     * @param isLargerBetter is the default metric better when larger or smaller
     * @param evaluateFn     evaluate function that returns one metric enclosed in a case class.
     *                       Note: dataset consists of two columns: (label, prediction)
     * @return a new regression evaluator
     */
    def custom(
      metricName: String,
      isLargerBetter: Boolean = true,
      evaluateFn: Dataset[(Double, Double)] => Double
    ): OpRegressionEvaluatorBase[SingleMetric] = {
      val islbt = isLargerBetter
      new OpRegressionEvaluatorBase[SingleMetric](
        uid = UID[OpRegressionEvaluatorBase[SingleMetric]]
      ) {
        override val name: String = metricName
        override val isLargerBetter: Boolean = islbt

        override def getDefaultMetric: SingleMetric => Double = _.value

        override def evaluateAll(dataset: Dataset[_]): SingleMetric = {
          // scalastyle:off
          import dataset.sparkSession.implicits._
          // scalastyle:on
          val ds = dataset.select(getLabelCol, getPredictionCol).as[(Double, Double)]
          val metric = evaluateFn(ds)
          SingleMetric(name, metric)
        }
      }
    }

  }

}

/**
 * A container for a single evaluation metric for evaluators
 *
 * @param name  metric name
 * @param value metric value
 */
case class SingleMetric(name: String, value: Double) extends EvaluationMetrics {
  override def toMap: Map[String, Any] = Map(name -> value)
  override def toString: String = JsonUtils.toJsonString(this.toMap, pretty = true)
}

/**
 * A container for multiple evaluation metrics for evaluators
 *
 * @param metrics map of evaluation metrics
 */
case class MultiMetrics(metrics: Map[String, EvaluationMetrics]) extends EvaluationMetrics {
  override def toMap: Map[String, Any] = metrics.flatMap {
    case (name, evalMetrics) => evalMetrics.toMap.map { case (k, v) => s"($name)_$k" -> v }
  }
  override def toString: String = JsonUtils.toJsonString(this.toMap, pretty = true)
}

