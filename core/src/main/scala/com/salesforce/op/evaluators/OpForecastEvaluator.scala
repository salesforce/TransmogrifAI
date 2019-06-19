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
 * @param name           name of default metric
 * @param isLargerBetter is metric better if larger
 * @param uid            uid for instance
 */

private[op] class OpForecastEvaluator
(
  override val name: EvalMetric = OpEvaluatorNames.Regression,
  override val isLargerBetter: Boolean = false,
  override val uid: String = UID[OpRegressionEvaluator]
) extends OpRegressionEvaluatorBase[ForecastMetrics](uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: ForecastMetrics => Double = _.sMAPE

  override def evaluateAll(data: Dataset[_]): ForecastMetrics = {
    val dataUse = makeDataToUse(data, getLabelCol)

    val smape: Double = getSMAPE(dataUse, getLabelCol, getPredictionValueCol)
    val metrics = ForecastMetrics(
      sMAPE = smape
    )

    log.info("Evaluated metrics: {}", metrics.toString)
    metrics

  }

  protected def getSMAPE(data: Dataset[_], labelCol: String, predictionValueCol: String): Double = {
    data.select(labelCol, predictionValueCol).rdd
      .map(r => ReduceSMAPE(r.getAs[Double](0), r.getAs[Double](1)))
      .reduce(_ + _).metric
  }
}

// https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
case class ReduceSMAPE(nominator: Double, denominator: Double, cnt: Long) {
  def +(that: ReduceSMAPE): ReduceSMAPE = {
    ReduceSMAPE(this.nominator + that.nominator, this.denominator + that.denominator, this.cnt + that.cnt)
  }

  def metric: Double = {
    if (denominator == 0.0) {
      Double.PositiveInfinity
    } else {
      (nominator / denominator) / cnt
    }
  }
}

object ReduceSMAPE {
  def apply(y: Double, y_hat: Double): ReduceSMAPE = {
    ReduceSMAPE(2 * Math.abs(y - y_hat), Math.abs(y) + Math.abs(y_hat), 1L)
  }
}

/**
 * Metrics of Regression Problem
 *
 * @param sMAPE symmetric Mean Absolute Percentage Error
 *
 */
case class ForecastMetrics
(
  sMAPE: Double
) extends EvaluationMetrics
