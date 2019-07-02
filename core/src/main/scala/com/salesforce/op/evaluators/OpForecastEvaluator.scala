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
import com.twitter.algebird.Operators._
import com.twitter.algebird.Semigroup
import com.twitter.algebird.macros.caseclass

import scala.collection.mutable

/**
 *
 * Instance to evaluate Forecast metrics
 * The metrics are SMAPE
 * Default evaluation returns SMAPE
 *
 * See: https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
 *
 * @param seasonalWindow length of the season (e.g. 7 for daily data with weekly seasonality)
 * @param maxItems       max number of items to process (default: 10 years of hourly data)
 * @param name           name of default metric
 * @param isLargerBetter is metric better if larger
 * @param uid            uid for instance
 */

private[op] class OpForecastEvaluator
(
  val seasonalWindow: Int = 1,
  val maxItems: Int = 87660,
  override val name: EvalMetric = OpEvaluatorNames.Forecast,
  override val isLargerBetter: Boolean = false,
  override val uid: String = UID[OpForecastEvaluator]
) extends OpRegressionEvaluatorBase[ForecastMetrics](uid) {

  require(seasonalWindow > 0, "seasonalWindow must not be negative")
  require(maxItems > 0, "maxItems must not be negative")

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: ForecastMetrics => Double = _.SMAPE

  override def evaluateAll(data: Dataset[_]): ForecastMetrics = {
    val dataUse = makeDataToUse(data, getLabelCol)

    val metrics = computeMetrics(dataUse, getLabelCol, getPredictionValueCol)
    log.info("Evaluated metrics: {}", metrics.toString)
    metrics

  }

  protected def computeMetrics(data: Dataset[_], labelCol: String, predictionValueCol: String): ForecastMetrics = {

    val rows = data.select(labelCol, predictionValueCol).rdd
      .map(r => (r.getAs[Double](0), r.getAs[Double](1))).take(maxItems)

    val cnt = rows.length
    val seasonalLimit = cnt - seasonalWindow

    var i = 0
    var (seasonalAbsDiff, absDiffSum, smapeSum) = (0.0, 0.0, 0.0)

    while (i < cnt) {
      val (y, yHat) = rows(i)
      val (ySeasonal, _) = rows(i + seasonalWindow)
      if (i < seasonalLimit) {
        seasonalAbsDiff += Math.abs(y - ySeasonal)
      }
      val absDiff = Math.abs(y - yHat)
      val sumAbs = Math.abs(y) + Math.abs(yHat)
      if (sumAbs > 0) {
        smapeSum += absDiff / sumAbs
      }

      absDiffSum += absDiff
      i += 1
    }

    val seasonalError = seasonalAbsDiff / seasonalLimit
    val maseDenominator = seasonalError * cnt

    ForecastMetrics(
      SMAPE = if (cnt > 0) 2 * smapeSum / cnt else 0.0,
      SeasonalError = seasonalError,
      MASE = if (maseDenominator > 0) absDiffSum / maseDenominator else 0.0
    )

  }
}

/**
 * Metrics of Forecasting Problem
 *
 * @param SMAPE         Symmetric Mean Absolute Percentage Error
 * @param SeasonalError Seasonal Error
 * @param MASE          Mean Absolute Scaled Error
 *
 */
case class ForecastMetrics(SMAPE: Double, SeasonalError: Double, MASE: Double) extends EvaluationMetrics
