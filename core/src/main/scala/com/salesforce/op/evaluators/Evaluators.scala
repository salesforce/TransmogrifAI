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
import org.apache.spark.sql.Dataset

class AuPR  extends OpBinaryClassificationEvaluator(name = BinaryClassEvalMetrics.AuPR, isLargerBetter = true) {
  override def evaluate(dataset: Dataset[_]): Double =
    getBinaryEvaluatorMetric(BinaryClassEvalMetrics.AuPR, dataset, default = 0.0)
}

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
    def brierScore(): OpBinScoreEvaluator = new OpBinScoreEvaluator()

    /**
     * Area under ROC
     */
    def auROC(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = BinaryClassEvalMetrics.AuROC, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getBinaryEvaluatorMetric(BinaryClassEvalMetrics.AuROC, dataset, default = 0.0)
      }




    /**
     * Area under Precision/Recall curve
     */
    def auPR(): OpBinaryClassificationEvaluator = new AuPR

    /**
     * Precision
     */
    def precision(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(
        name = BinaryClassEvalMetrics.Precision, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getBinaryEvaluatorMetric(BinaryClassEvalMetrics.Precision, dataset, default = 0.0)
      }


    /**
     * Recall
     */
    def recall(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = BinaryClassEvalMetrics.Recall, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getBinaryEvaluatorMetric(BinaryClassEvalMetrics.Recall, dataset, default = 0.0)
      }

    /**
     * F1 score
     */
    def f1(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = BinaryClassEvalMetrics.F1, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getBinaryEvaluatorMetric(BinaryClassEvalMetrics.F1, dataset, default = 0.0)
      }

    /**
     * Prediction error
     */
    def error(): OpBinaryClassificationEvaluator =
      new OpBinaryClassificationEvaluator(name = BinaryClassEvalMetrics.Error, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          1.0 - getBinaryEvaluatorMetric(BinaryClassEvalMetrics.Error, dataset, default = 1.0)
      }

    /**
     * Custom binary evaluator
     *
     * @param metricName     name of default metric
     * @param isLargerBetter is the default metric better when larger or smaller
     * @param evaluateFn     evaluate function:
     *                       - input: dataset consisting of four columns:
     *                       (label, raw prediction, probability, prediction)
     *                       - output: a single metric value
     *                       Note: it the user's responsibility to take care of all the error scenarios in evaluateFn
     * @return a binary evaluator
     */
    def custom(
      metricName: String,
      isLargerBetter: Boolean = true,
      evaluateFn: Dataset[(Double, OPVector#Value, OPVector#Value, Double)] => Double
    ): OpBinaryClassificationEvaluatorBase[SingleMetric] = {
      val largerBetter = isLargerBetter
      new OpBinaryClassificationEvaluatorBase[SingleMetric](
        uid = UID[OpBinaryClassificationEvaluatorBase[SingleMetric]]
      ) {
        override val name: EvalMetric = OpEvaluatorNames.Custom(metricName, metricName)
        override val isLargerBetter: Boolean = largerBetter
        override def getDefaultMetric: SingleMetric => Double = _.value

        override def evaluateAll(dataset: Dataset[_]): SingleMetric = {
          import dataset.sparkSession.implicits._
          val dataUse = makeDataToUse(dataset, getLabelCol)
          val ds = dataUse
            .select(getLabelCol, getRawPredictionCol, getProbabilityCol, getPredictionValueCol)
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
      new OpMultiClassificationEvaluator(name = MultiClassEvalMetrics.Precision, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getMultiEvaluatorMetric(MultiClassEvalMetrics.Precision, dataset, default = 0.0)
      }

    /**
     * Weighted Recall
     */
    def recall(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = MultiClassEvalMetrics.Recall, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getMultiEvaluatorMetric(MultiClassEvalMetrics.Recall, dataset, default = 0.0)
      }

    /**
     * F1 Score
     */
    def f1(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = MultiClassEvalMetrics.F1, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getMultiEvaluatorMetric(MultiClassEvalMetrics.F1, dataset, default = 0.0)
      }

    /**
     * Prediction Error
     */
    def error(): OpMultiClassificationEvaluator =
      new OpMultiClassificationEvaluator(name = MultiClassEvalMetrics.Error, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          1.0 - getMultiEvaluatorMetric(MultiClassEvalMetrics.Error, dataset, default = 1.0)
      }

    /**
     * Custom multiclass evaluator
     *
     * @param metricName     name of default metric
     * @param isLargerBetter is the default metric better when larger or smaller
     * @param evaluateFn     evaluate function:
     *                       - input: dataset consisting of four columns:
     *                       (label, raw prediction, probability, prediction)
     *                       - output: a single metric value
     *                       Note: it the user's responsibility to take care of all the error scenarios in evaluateFn
     * @return a new multiclass evaluator
     */
    def custom(
      metricName: String,
      isLargerBetter: Boolean = true,
      evaluateFn: Dataset[(Double, OPVector#Value, OPVector#Value, Double)] => Double
    ): OpMultiClassificationEvaluatorBase[SingleMetric] = {
      val largerBetter = isLargerBetter
      new OpMultiClassificationEvaluatorBase[SingleMetric](
        uid = UID[OpMultiClassificationEvaluatorBase[SingleMetric]]
      ) {
        override val name: EvalMetric = OpEvaluatorNames.Custom(metricName, metricName)
        override val isLargerBetter: Boolean = largerBetter
        override def getDefaultMetric: SingleMetric => Double = _.value

        override def evaluateAll(dataset: Dataset[_]): SingleMetric = {
          import dataset.sparkSession.implicits._
          val dataUse = makeDataToUse(dataset, getLabelCol)
          val ds = dataUse
            .select(getLabelCol, getRawPredictionCol, getProbabilityCol, getPredictionValueCol)
            .as[(Double, OPVector#Value, OPVector#Value, Double)]
          val metric = evaluateFn(ds)
          SingleMetric(name.humanFriendlyName, metric)
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
      new OpRegressionEvaluator(name = RegressionEvalMetrics.MeanSquaredError, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.MeanSquaredError, dataset, default = 0.0)
      }

    /**
     * Mean Absolute Error
     */
    def mae(): OpRegressionEvaluator =
      new OpRegressionEvaluator(name = RegressionEvalMetrics.MeanAbsoluteError, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.MeanAbsoluteError, dataset, default = 0.0)
      }

    /**
     * R2
     */
    def r2(): OpRegressionEvaluator =
      new OpRegressionEvaluator(name = RegressionEvalMetrics.R2, isLargerBetter = true) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.R2, dataset, default = 0.0)
      }

    /**
     * Root Mean Squared Error
     */
    def rmse(): OpRegressionEvaluator =
      new OpRegressionEvaluator(name = RegressionEvalMetrics.RootMeanSquaredError, isLargerBetter = false) {
        override def evaluate(dataset: Dataset[_]): Double =
          getRegEvaluatorMetric(RegressionEvalMetrics.RootMeanSquaredError, dataset, default = 0.0)
      }

    /**
     * Custom regression evaluator
     *
     * @param metricName     name of default metric
     * @param isLargerBetter is the default metric better when larger or smaller
     * @param evaluateFn     evaluate function:
     *                       - input: dataset consisting of two columns: (label, prediction).
     *                       - output: a single metric value
     *                       Note: it the user's responsibility to take care of all the error scenarios in evaluateFn
     * @return a new regression evaluator
     */
    def custom(
      metricName: String,
      isLargerBetter: Boolean = true,
      evaluateFn: Dataset[(Double, Double)] => Double
    ): OpRegressionEvaluatorBase[SingleMetric] = {
      val largerBetter = isLargerBetter
      new OpRegressionEvaluatorBase[SingleMetric](
        uid = UID[OpRegressionEvaluatorBase[SingleMetric]]
      ) {
        override val name: EvalMetric = OpEvaluatorNames.Custom(metricName, metricName)
        override val isLargerBetter: Boolean = largerBetter
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

