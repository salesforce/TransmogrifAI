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


case class AllRegressionMetrics(RootMeanSquaredError: Double,
  MeanSquaredError: Double,
  R2: Double,
  MeanAbsoluteError: Double,
  ByMeanNRMSE: Double, BySDNRMSE: Double, ByMedianNRMSE: Double, ByRangeNRMSE: Double,
  ByIQRNRMSE: Double) extends EvaluationMetrics


case class NRMSEMetrics(ByMeanNRMSE: Double, BySDNRMSE: Double, ByMedianNRMSE: Double, ByRangeNRMSE: Double,
  ByIQRNRMSE: Double) extends EvaluationMetrics

class NRMSEEvaluator extends OpRegressionEvaluatorBase[NRMSEMetrics](uid = UID[NRMSEEvaluator]) {
  def getDefaultMetric: NRMSEMetrics => Double = _.ByMeanNRMSE

  override def name: EvalMetric = EvalMetric.withNameInsensitive(
    "normalized root mean squared error",
    false)

  def divideBy(rmse: Double, withValue: Double): Double = if (withValue != 0.0) rmse / withValue else rmse

  override def evaluateAll(dataset: Dataset[_]): NRMSEMetrics = {
    val rmse = Evaluators.Regression.rmse().setPredictionCol(getPredictionCol).setLabelCol(getLabelCol)
      .evaluate(dataset)
    import dataset.sqlContext.implicits._
    val labelDS = dataset.select(getLabelCol).as[Double]
    val summary = labelDS.summary("mean", "stddev", "min", "max", "25%", "50%", "75%")
    val (mean, sd, range, median, iqr) = {
      val summaryCol = summary.select(getLabelCol).as[String].map(_.toDouble).as[Double].collect()
      (summaryCol(0), summaryCol(1), summaryCol(3) - summaryCol(2), summaryCol(5), summaryCol(6) - summaryCol(4))
    }
    val nmrseMetrics = new NRMSEMetrics(
      ByMeanNRMSE = divideBy(rmse, mean),
      BySDNRMSE = divideBy(rmse, sd),
      ByMedianNRMSE = divideBy(rmse, median),
      ByRangeNRMSE = divideBy(rmse, range),
      ByIQRNRMSE = divideBy(rmse, iqr)
    )
    println(s"NMRSE metrics ${nmrseMetrics}")
    nmrseMetrics
  }
}

class AllRegressionEvaluator extends OpRegressionEvaluatorBase[AllRegressionMetrics](
  uid = UID[AllRegressionEvaluator]) {
  def getDefaultMetric: AllRegressionMetrics => Double = _.RootMeanSquaredError

  override def name: EvalMetric = EvalMetric.withNameInsensitive("all regression",
    false)

  override def evaluateAll(dataset: Dataset[_]): AllRegressionMetrics = {
    val regMetrics = Evaluators.Regression().setPredictionCol(getPredictionCol).setLabelCol(getLabelCol)
      .evaluateAll(dataset)
    val nrmseMetrics = new NRMSEEvaluator().setPredictionCol(getPredictionCol).setLabelCol(getLabelCol)
      .evaluateAll(dataset)
    val finalMetrics = new AllRegressionMetrics(RootMeanSquaredError = regMetrics.RootMeanSquaredError,
      MeanAbsoluteError = regMetrics.MeanAbsoluteError,
      MeanSquaredError = regMetrics.MeanAbsoluteError,
      R2 = regMetrics.R2,
      ByMeanNRMSE = nrmseMetrics.ByMeanNRMSE,
      BySDNRMSE = nrmseMetrics.BySDNRMSE,
      ByMedianNRMSE = nrmseMetrics.ByMedianNRMSE,
      ByRangeNRMSE = nrmseMetrics.ByRangeNRMSE,
      ByIQRNRMSE = nrmseMetrics.ByIQRNRMSE)
    println(s"final metrics ${finalMetrics}")
    finalMetrics
  }
}

class MeanNRMSE extends NRMSEEvaluator {
  override def name: EvalMetric = EvalMetric.withNameInsensitive("mean normalized root mean squared error",
    false)

  override def getDefaultMetric: NRMSEMetrics => Double = _.ByMeanNRMSE
}

class SDNRMSE extends NRMSEEvaluator {
  override def name: EvalMetric = EvalMetric.withNameInsensitive("standard deviation normalized root" +
    " mean squared error", false)

  override def getDefaultMetric: NRMSEMetrics => Double = _.BySDNRMSE
}

class RangeNRMSE extends NRMSEEvaluator {
  override def name: EvalMetric = EvalMetric.withNameInsensitive("range normalized root" +
    " mean squared error", false)

  override def getDefaultMetric: NRMSEMetrics => Double = _.ByRangeNRMSE
}

class MedianNRMSE extends NRMSEEvaluator {
  override def name: EvalMetric = EvalMetric.withNameInsensitive("median normalized root" +
    " mean squared error", false)

  override def getDefaultMetric: NRMSEMetrics => Double = _.ByMedianNRMSE
}

class IQRNRMSE extends NRMSEEvaluator {
  override def name: EvalMetric = EvalMetric.withNameInsensitive("inter quantile range normalized root" +
    " mean squared error", false)

  override def getDefaultMetric: NRMSEMetrics => Double = _.ByIQRNRMSE
}
