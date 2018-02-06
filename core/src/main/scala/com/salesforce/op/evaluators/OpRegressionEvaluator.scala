/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.evaluators

import com.salesforce.op.UID
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.Dataset
import com.salesforce.op.features.types._
import org.slf4j.LoggerFactory
import enumeratum._

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
  override val name: String = OpEvaluatorNames.regression,
  override val isLargerBetter: Boolean = false,
  override val uid: String = UID[OpRegressionEvaluator]
) extends OpRegressionEvaluatorBase[RegressionMetrics](uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  def getDefaultMetric: RegressionMetrics => Double = _.RootMeanSquaredError

  override def evaluateAll(data: Dataset[_]): RegressionMetrics = {

    val rmse = getRegEvaluatorMetric(RegressionEvalMetrics.RootMeanSquaredError, data)
    val mse = getRegEvaluatorMetric(RegressionEvalMetrics.MeanSquaredError, data)
    val r2 = getRegEvaluatorMetric(RegressionEvalMetrics.R2, data)
    val mae = getRegEvaluatorMetric(RegressionEvalMetrics.MeanAbsoluteError, data)

    val metrics = RegressionMetrics(
      RootMeanSquaredError = rmse, MeanSquaredError = mse, R2 = r2, MeanAbsoluteError = mae
    )

    log.info("Evaluated metrics: {}", metrics.toString)
    metrics

  }

  final private[op] def getRegEvaluatorMetric(metricName: RegressionEvalMetric, dataset: Dataset[_]): Double = {
    new RegressionEvaluator()
      .setLabelCol(getLabelCol)
      .setPredictionCol(getPredictionCol)
      .setMetricName(metricName.sparkEntryName)
      .evaluate(dataset)
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

/* Regression Metrics */
sealed abstract class RegressionEvalMetric(val sparkEntryName: String) extends EnumEntry with Serializable
object RegressionEvalMetrics extends Enum[RegressionEvalMetric] {
  val values: Seq[RegressionEvalMetric] = findValues
  case object RootMeanSquaredError extends RegressionEvalMetric("rmse")
  case object MeanSquaredError extends RegressionEvalMetric("mse")
  case object R2 extends RegressionEvalMetric("r2")
  case object MeanAbsoluteError extends RegressionEvalMetric("mae")
}
