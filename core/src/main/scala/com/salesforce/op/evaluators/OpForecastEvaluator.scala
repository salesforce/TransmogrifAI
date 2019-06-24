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
 * @param name           name of default metric
 * @param isLargerBetter is metric better if larger
 * @param uid            uid for instance
 */

private[op] class OpForecastEvaluator
(
  val seasonalWindow: Int = 1,
  override val name: EvalMetric = OpEvaluatorNames.Forecast,
  override val isLargerBetter: Boolean = false,
  override val uid: String = UID[OpForecastEvaluator]
) extends OpRegressionEvaluatorBase[ForecastMetrics](uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: ForecastMetrics => Double = _.SMAPE

  override def evaluateAll(data: Dataset[_]): ForecastMetrics = {
    val dataUse = makeDataToUse(data, getLabelCol)

    val metrics = getMetrics(dataUse, getLabelCol, getPredictionValueCol)
    log.info("Evaluated metrics: {}", metrics.toString)
    metrics

  }


  protected def getMetrics(data: Dataset[_], labelCol: String, predictionValueCol: String): ForecastMetrics = {
    val res = data.select(labelCol, predictionValueCol).rdd
      .map(r => MetricValue(r.getAs[Double](0), r.getAs[Double](1))).coalesce(1)
      .fold(MetricValue.zero(seasonalWindow))(_ + _)
    ForecastMetrics(
      SMAPE = res.smape,
      seasonalError = res.seasonalError,
      MASE = res.mase
    )

  }
}

// https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
object MetricValue {
  def apply(y: Double, yHat: Double): MetricValue = {
    val absDiff = Math.abs(y - yHat)
    val sumAbs = Math.abs(y) + Math.abs(yHat)
    val smapeSum = if (sumAbs > 0) {
      absDiff / sumAbs
    } else {
      0.0
    }
    MetricValue(absDiff, smapeSum, SeasonalAbsDiff.apply(0, y), 1L)
  }

  def zero(seasonalWindow: Int): MetricValue = {
    MetricValue(0d, 0d, SeasonalAbsDiff.zero(seasonalWindow), 0L)
  }

  implicit val smapeSG: Semigroup[MetricValue] = caseclass.semigroup[MetricValue]
}

case class MetricValue private(absDiff: Double, smapeSum: Double, seasonalAbsDiff: SeasonalAbsDiff, cnt: Long) {
  def smape: Double = {
    if (cnt == 0) {
      Double.NaN
    } else {
      2 * smapeSum / cnt
    }
  }

  def seasonalError: Double = seasonalAbsDiff.seasonalError(cnt)

  def mase: Double = absDiff / (seasonalError * cnt)

}

object SeasonalAbsDiff {
  def apply(seasonalWindow: Int, y: Double): SeasonalAbsDiff =
    SeasonalAbsDiff(seasonalWindow, 1, mutable.ListBuffer[Double](y), 0.0)

  def zero(seasonalWindow: Int): SeasonalAbsDiff =
    SeasonalAbsDiff(seasonalWindow, 0, mutable.ListBuffer[Double](), 0.0)

  implicit val seasonalAbsDiffSG: Semigroup[SeasonalAbsDiff] = new Semigroup[SeasonalAbsDiff] {
    override def plus(x: SeasonalAbsDiff, y: SeasonalAbsDiff): SeasonalAbsDiff = x.combine(y)
  }
}

case class SeasonalAbsDiff private
(seasonalWindow: Int, lagsLength: Int, lags: mutable.ListBuffer[Double], absDiff: Double) {
  def combine(that: SeasonalAbsDiff): SeasonalAbsDiff = {
    var newLength = lagsLength + that.lagsLength
    var l = this.lags
    var newAbsDiff = this.absDiff + that.absDiff
    var idx = seasonalWindow - lagsLength

    while (newLength > seasonalWindow) {
      newAbsDiff += Math.abs(l.head - that.lags(idx))
      l = l.tail
      newLength -= 1
      idx += 1
    }
    SeasonalAbsDiff(seasonalWindow, newLength, l ++ that.lags, newAbsDiff)
  }

  def seasonalError(dataLength: Long): Double = {
    val denominator = dataLength - seasonalWindow
    if (denominator > 0) {
      absDiff / denominator
    } else {
      Double.NaN
    }
  }
}


/**
 * Metrics of Forecasting Problem
 *
 * @param SMAPE symmetric Mean Absolute Percentage Error
 *
 */
case class ForecastMetrics(SMAPE: Double, seasonalError: Double, MASE: Double) extends EvaluationMetrics
