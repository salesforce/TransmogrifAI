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

    /*
     * Brier Score for the prediction
     */
    def brierScore(): OpBinScoreEvaluator =
      new OpBinScoreEvaluator(name = BinaryClassEvalMetrics.brierScore, isLargerBetter = true)

    /**
     * Area under ROC
     */
    def auROC(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(
        name = BinaryClassEvalMetrics.AuROC, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getBinaryEvaluatorMetric(BinaryClassEvalMetrics.AuROC, dataset)
      }

    /**
     * Area under Precision/Recall curve
     */
    def auPR(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = BinaryClassEvalMetrics.AuPR, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getBinaryEvaluatorMetric(BinaryClassEvalMetrics.AuPR, dataset)
      }

    /**
     * Precision
     */
    def precision(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(
        name = MultiClassEvalMetrics.Precision, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double = {
          import dataset.sparkSession.implicits._
          val dataUse = makeDataToUse(dataset, getLabelCol)
          new MulticlassMetrics(dataUse.select(getPredictionValueCol, getLabelCol).as[(Double, Double)].rdd)
            .precision(1.0)
        }
      }

    /**
     * Recall
     */
    def recall(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(
        name = MultiClassEvalMetrics.Recall, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double = {
          import dataset.sparkSession.implicits._
          val dataUse = makeDataToUse(dataset, getLabelCol)
          new MulticlassMetrics(dataUse.select(getPredictionValueCol, getLabelCol).as[(Double, Double)].rdd)
            .recall(1.0)
        }
      }

    /**
     * F1 score
     */
    def f1(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = MultiClassEvalMetrics.F1, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double = {
          import dataset.sparkSession.implicits._
          val dataUse = makeDataToUse(dataset, getLabelCol)
          new MulticlassMetrics(
            dataUse.select(getPredictionValueCol, getLabelCol).as[(Double, Double)].rdd)
            .fMeasure(1.0)
        }
      }

    /**
     * Prediction error
     */
    def error(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(
        name = MultiClassEvalMetrics.Error, isLargerBetter = false) {
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
        override val name: EvalMetric = OpEvaluatorNames.Custom(metricName, metricName)
        override val isLargerBetter: Boolean = islbt
        override def getDefaultMetric: SingleMetric => Double = _.value
        override def evaluateAll(dataset: Dataset[_]): SingleMetric = {
          import dataset.sparkSession.implicits._
          val dataUse = makeDataToUse(dataset, getLabelCol)
          val ds = dataUse.select(getLabelCol, getRawPredictionCol, getProbabilityCol, getPredictionValueCol)
            .as[(Double, OPVector#Value, OPVector#Value, Double)]
          val metric = evaluateFn(ds)
          SingleMetric(name.humanFriendlyName, metric)
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
      new OpMultiClassificationEvaluator(
        name = MultiClassEvalMetrics.Precision, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getMultiEvaluatorMetric(MultiClassEvalMetrics.Precision, dataset)
      }

    /**
     * Weighted Recall
     */
    def recall(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = MultiClassEvalMetrics.Recall, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getMultiEvaluatorMetric(MultiClassEvalMetrics.Recall, dataset)
      }

    /**
     * F1 Score
     */
    def f1(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = MultiClassEvalMetrics.F1, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getMultiEvaluatorMetric(MultiClassEvalMetrics.F1, dataset)
      }

    /**
     * Prediction Error
     */
    def error(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = MultiClassEvalMetrics.Error, isLargerBetter = false) {
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
        override val name: EvalMetric = OpEvaluatorNames.Custom(metricName, metricName)
        override val isLargerBetter: Boolean = islbt

        override def getDefaultMetric: SingleMetric => Double = _.value

        override def evaluateAll(dataset: Dataset[_]): SingleMetric = {
          import dataset.sparkSession.implicits._
          val dataUse = makeDataToUse(dataset, getLabelCol)
          val ds = dataUse.select(getLabelCol, getRawPredictionCol, getProbabilityCol, getPredictionValueCol)
            .as[(Double, OPVector#Value, OPVector#Value, Double)]
          try {
            val metric = evaluateFn(ds)
            SingleMetric(name.humanFriendlyName, metric)
          } catch {
            case iae: IllegalArgumentException =>
              val size = dataset.count
              val desc = s"dataset with ($getLabelCol, $getRawPredictionCol," +
                s" $getProbabilityCol, $getPredictionValueCol)"
              val msg = if (size == 0) s"empty $desc" else s"$desc of $size rows"
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
      new OpRegressionEvaluator(
        name = RegressionEvalMetrics.MeanSquaredError, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.MeanSquaredError, dataset)
      }

    /**
     * Mean Absolute Error
     */
    def mae(): OpRegressionEvaluator =
      new OpRegressionEvaluator(
        name = RegressionEvalMetrics.MeanAbsoluteError, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.MeanAbsoluteError, dataset)
      }

    /**
     * R2
     */
    def r2(): OpRegressionEvaluator =
      new OpRegressionEvaluator(name = RegressionEvalMetrics.R2, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.R2, dataset)
      }

    /**
     * Root Mean Squared Error
     */
    def rmse(): OpRegressionEvaluator =
      new OpRegressionEvaluator(
        name = RegressionEvalMetrics.RootMeanSquaredError, isLargerBetter = false) {
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
        override val name: EvalMetric = OpEvaluatorNames.Custom(metricName, metricName)
        override val isLargerBetter: Boolean = islbt

        override def getDefaultMetric: SingleMetric => Double = _.value

        override def evaluateAll(dataset: Dataset[_]): SingleMetric = {
          import dataset.sparkSession.implicits._
          val dataUse = makeDataToUse(dataset, getLabelCol)
          val ds = dataUse.select(getLabelCol, getPredictionValueCol).as[(Double, Double)]
          val metric = evaluateFn(ds)
          SingleMetric(name.humanFriendlyName, metric)
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
