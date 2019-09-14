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
import com.salesforce.op.utils.spark.RichEvaluator._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.Dataset
import org.slf4j.LoggerFactory

/**
 *
 * Instance to evaluate Regression metrics
 * The metrics are rmse, mse, r2 and mae
 * Default evaluation returns Root Mean Squared Error
 *
 * @param name name of default metric
 * @param isLargerBetter is metric better if larger
 * @param uid uid for instance
 */

private[op] class OpRegressionEvaluator
(
  override val name: EvalMetric = OpEvaluatorNames.Regression,
  override val uid: String = UID[OpRegressionEvaluator]
) extends OpRegressionEvaluatorBase[RegressionMetrics](uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: RegressionMetrics => Double = _.RootMeanSquaredError

  override def evaluateAll(data: Dataset[_]): RegressionMetrics = {
    val dataUse = makeDataToUse(data, getLabelCol)
    val rmse = getRegEvaluatorMetric(RegressionEvalMetrics.RootMeanSquaredError, dataUse, default = 0.0)
    val mse = getRegEvaluatorMetric(RegressionEvalMetrics.MeanSquaredError, dataUse, default = 0.0)
    val r2 = getRegEvaluatorMetric(RegressionEvalMetrics.R2, dataUse, default = 0.0)
    val mae = getRegEvaluatorMetric(RegressionEvalMetrics.MeanAbsoluteError, dataUse, default = 0.0)

    val metrics = RegressionMetrics(
      RootMeanSquaredError = rmse, MeanSquaredError = mse, R2 = r2, MeanAbsoluteError = mae
    )

    log.info("Evaluated metrics: {}", metrics.toString)
    metrics

  }

  final protected def getRegEvaluatorMetric(
    metricName: RegressionEvalMetric,
    dataset: Dataset[_],
    default: => Double
  ): Double = {
    val labelName = getLabelCol
    val dataUse = makeDataToUse(dataset, labelName)
    new RegressionEvaluator()
      .setLabelCol(labelName)
      .setPredictionCol(getPredictionValueCol)
      .setMetricName(metricName.sparkEntryName)
      .evaluateOrDefault(dataUse, default = default)
  }
}


/**
 * Metrics of Regression Problem
 *
 * @param RootMeanSquaredError
 * @param MeanSquaredError
 * @param R2
 * @param MeanAbsoluteError
 */
case class RegressionMetrics
(
  RootMeanSquaredError: Double,
  MeanSquaredError: Double,
  R2: Double,
  MeanAbsoluteError: Double
) extends EvaluationMetrics
