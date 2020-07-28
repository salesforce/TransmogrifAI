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
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.{DoubleArrayParam, DoubleParam}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.slf4j.LoggerFactory

/**
 * Evaluator for regression metrics.
 * The metrics are RMSE, MSE, R2, MASE, and a histogram of the signed percentage errors.
 * For the percentage errors, it deals with the difficulties that occur with label
 * values around 0, and exposes several parameters to control that behavior.
 * Default evaluation returns Root Mean Squared Error
 *
 * @param name name of default metric
 * @param uid uid for instance
 */

private[op] class OpRegressionEvaluator
(
  override val name: EvalMetric = OpEvaluatorNames.Regression,
  override val uid: String = UID[OpRegressionEvaluator]
) extends OpRegressionEvaluatorBase[RegressionMetrics](uid) {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  final val signedPercentageErrorHistogramBins = new DoubleArrayParam(
    parent = this,
    name = "signedPercentageErrorHistogramBins",
    doc = "sequence of bins for signed percentage error histogram",
    isValid = l => l.nonEmpty && (l sameElements l.sorted)
  )
  setDefault(signedPercentageErrorHistogramBins,
    Array(Double.NegativeInfinity) ++ (-100.0 to 100.0 by 10) ++ Array(Double.PositiveInfinity)
  )

  def setPercentageErrorHistogramBins(v: Array[Double]): this.type = set(signedPercentageErrorHistogramBins, v)

  final val scaledErrorCutoff = new DoubleParam(
    parent = this,
    name = "scaledErrorCutoff",
    doc = "label value cutoff below which percentage error is implemented as a scaled error " +
      "with a fixed denominator to avoid problems with label values around 0",
    isValid = (d: Double) => d > 0.0
  )
  setDefault(scaledErrorCutoff, 1E-3)

  def setScaledErrorCutoff(v: Double): this.type = set(scaledErrorCutoff, v)
  def getScaledErrorCutoff: Option[Double] = get(scaledErrorCutoff)

  final val smartCutoffRatio = new DoubleParam(
    parent = this,
    name = "smartCutoffRatio",
    doc = "if set, scaledErrorCutoff is determined smartly by taking the average absolute magnitude " +
      "of the data multiplied with this ratio",
    isValid = (d: Double) => d > 0.0
  )

  def setSmartCutoffRatio(v: Double): this.type = set(smartCutoffRatio, v)

  def getDefaultMetric: RegressionMetrics => Double = _.RootMeanSquaredError

  override def evaluateAll(data: Dataset[_]): RegressionMetrics = {
    val dataUse = makeDataToUse(data, getLabelCol)
    val rmse = getRegEvaluatorMetric(RegressionEvalMetrics.RootMeanSquaredError, dataUse, default = 0.0)
    val mse = getRegEvaluatorMetric(RegressionEvalMetrics.MeanSquaredError, dataUse, default = 0.0)
    val r2 = getRegEvaluatorMetric(RegressionEvalMetrics.R2, dataUse, default = 0.0)
    val mae = getRegEvaluatorMetric(RegressionEvalMetrics.MeanAbsoluteError, dataUse, default = 0.0)

    val histogram = calculateSignedPercentageErrorHistogram(dataUse)

    val metrics = RegressionMetrics(
      RootMeanSquaredError = rmse,
      MeanSquaredError = mse,
      R2 = r2,
      MeanAbsoluteError = mae,
      SignedPercentageErrorHistogram = SignedPercentageErrorHistogram(
        bins = $(signedPercentageErrorHistogramBins).toArray,
        counts = histogram
      )
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

  /**
   * Gets the histogram of the signed percentage errors
   *
   * @param data Data to use
   * @return Sequence of bin counts of the histogram
   */
  private def calculateSignedPercentageErrorHistogram(data: Dataset[_]): Array[Long] = {
    // Prep data
    val predictionsAndLabels = data
      .select(col(getPredictionValueCol).cast(DoubleType), col(getLabelCol).cast(DoubleType))
      .rdd
      .map { case Row(prediction: Double, label: Double) => (prediction, label) }

    // If we need to set the scaledErrorCutoff smartly, use the label data for that
    if (isDefined(smartCutoffRatio)) {
      val cutoff = calculateSmartCutoff(predictionsAndLabels)
      log.info(s"Smart scaledErrorCutoff was determined to be: $cutoff")
      setScaledErrorCutoff(cutoff)
    }

    val errors: RDD[Double] = predictionsAndLabels
      .map(x => calculateSignedPercentageError(x._1, x._2, $(scaledErrorCutoff)))
    errors.histogram($(signedPercentageErrorHistogramBins))
  }

  /**
   * Smartly calculates the scaledErrorCutoff
   *
   * @param predictionAndLabels  Data set containing predictions and labels
   * @return Suggested cutoff level for scaledErrorCutoff
   */
  protected def calculateSmartCutoff(predictionAndLabels: RDD[(Double, Double)]): Double = {
    val meanAbsoluteLabel = predictionAndLabels.map(_._2.abs).mean()
    // Take the max with scaledErrorCutoff to avoid a cutoff of 0 if labels are all 0.
    ($(smartCutoffRatio) * meanAbsoluteLabel) max $(scaledErrorCutoff)
  }

  /**
   * Calculates the signed percentage error, with cutoff logic to avoid large results
   * due to division by small numbers.
   *
   * @param prediction Predicted value
   * @param label      Actual value
   * @return Signed percentage error
   */
  private def calculateSignedPercentageError(
    prediction: Double,
    label: Double,
    scaledErrorCutoff: Double
  ): Double = {
    100.0 * (prediction - label) / (label.abs max scaledErrorCutoff)
  }

}


/**
 * Metrics of Regression Problem
 *
 * @param RootMeanSquaredError
 * @param MeanSquaredError
 * @param R2
 * @param MeanAbsoluteError
 * @param SignedPercentageErrorHistogram
 */
case class RegressionMetrics
(
  RootMeanSquaredError: Double,
  MeanSquaredError: Double,
  R2: Double,
  MeanAbsoluteError: Double,
  SignedPercentageErrorHistogram: SignedPercentageErrorHistogram
) extends EvaluationMetrics


/**
 * Histogram of signed percentage errors
 *
 * @param bins Histogram bins, where for example [-1, 0, 1] refer to bins [-1, 0), [0, 1]
 * @param counts Histogram counts (length of bins parameter - 1)
 */
case class SignedPercentageErrorHistogram
(
  @JsonDeserialize(contentAs = classOf[java.lang.Double])
  bins: Seq[Double],
  @JsonDeserialize(contentAs = classOf[java.lang.Long])
  counts: Seq[Long]
) extends EvaluationMetrics
