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

import com.salesforce.op.utils.json.{JsonLike, JsonUtils}
import com.salesforce.op.utils.spark.RichMetadata._
import enumeratum.{Enum, EnumEntry}
import org.apache.spark.sql.types.Metadata


/**
 * Trait for all different kinds of evaluation metrics
 */
trait EvaluationMetrics extends JsonLike {

  /**
   * Convert metrics class to a map
   * @return a map from metric name to metric value
   */
  def toMap: Map[String, Any] = JsonUtils.toMap(JsonUtils.toJsonTree(this))

  /**
   * Convert metrics into [[Metadata]] for saving
   *
   * @param skipUnsupported skip unsupported values
   * @throws RuntimeException in case of unsupported value type
   * @return [[Metadata]] metadata
   */
  def toMetadata(skipUnsupported: Boolean = false): Metadata = this.toMap.toMetadata(skipUnsupported)

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

/**
 * Eval metric
 */
sealed trait EvalMetric extends EnumEntry with Serializable {

  /**
   * Spark metric name
   */
  def sparkEntryName: String

  /**
   * Human friendly metric name
   */
  def humanFriendlyName: String

  /**
   * Is this metric being larger better or is smaller better
   */
  def isLargerBetter: Boolean

}

/**
 * Eval metric companion object
 */
object EvalMetric {

  def withNameInsensitive(name: String, isLargerBetter: Boolean = true): EvalMetric = {
    BinaryClassEvalMetrics.withNameInsensitiveOption(name)
      .orElse(MultiClassEvalMetrics.withNameInsensitiveOption(name))
      .orElse(RegressionEvalMetrics.withNameInsensitiveOption(name))
      .orElse(ForecastEvalMetrics.withNameInsensitiveOption(name))
      .orElse(OpEvaluatorNames.withNameInsensitiveOption(name))
      .getOrElse(OpEvaluatorNames.Custom(name, name, isLargerBetter))
  }
}

/**
 * Classification Metrics
 */
sealed abstract class ClassificationEvalMetric
(
  val sparkEntryName: String,
  val humanFriendlyName: String,
  val isLargerBetter: Boolean
) extends EvalMetric

/**
 * Binary Classification Metrics
 */
object BinaryClassEvalMetrics extends Enum[ClassificationEvalMetric] {
  val values = findValues
  case object Precision extends ClassificationEvalMetric("weightedPrecision", "precision", true)
  case object Recall extends ClassificationEvalMetric("weightedRecall", "recall", true)
  case object F1 extends ClassificationEvalMetric("f1", "f1", true)
  case object Error extends ClassificationEvalMetric("accuracy", "error", false)
  case object AuROC extends ClassificationEvalMetric("areaUnderROC", "area under ROC", true)
  case object AuPR extends ClassificationEvalMetric("areaUnderPR", "area under precision-recall", true)
  case object TP extends ClassificationEvalMetric("TP", "true positive", true)
  case object TN extends ClassificationEvalMetric("TN", "true negative", true)
  case object FP extends ClassificationEvalMetric("FP", "false positive", false)
  case object FN extends ClassificationEvalMetric("FN", "false negative", false)
  case object BrierScore extends ClassificationEvalMetric("brierScore", "brier score", false)
}

/**
 * Multi Classification Metrics
 */
object MultiClassEvalMetrics extends Enum[ClassificationEvalMetric] {
  val values = findValues
  case object Precision extends ClassificationEvalMetric("weightedPrecision", "precision", true)
  case object Recall extends ClassificationEvalMetric("weightedRecall", "recall", true)
  case object F1 extends ClassificationEvalMetric("f1", "f1", true)
  case object Error extends ClassificationEvalMetric("accuracy", "error", false)
  case object ThresholdMetrics extends ClassificationEvalMetric("thresholdMetrics", "threshold metrics", true)
}


/**
 * Regression Metrics
 */
sealed abstract class RegressionEvalMetric
(
  val sparkEntryName: String,
  val humanFriendlyName: String,
  val isLargerBetter: Boolean
) extends EvalMetric

/**
 * Regression Metrics
 */
object RegressionEvalMetrics extends Enum[RegressionEvalMetric] {
  val values: Seq[RegressionEvalMetric] = findValues
  case object RootMeanSquaredError extends RegressionEvalMetric("rmse", "root mean square error", false)
  case object MeanSquaredError extends RegressionEvalMetric("mse", "mean square error", false)
  case object R2 extends RegressionEvalMetric("r2", "r2", true)
  case object MeanAbsoluteError extends RegressionEvalMetric("mae", "mean absolute error", false)
  case object SignedPercentageErrorHistogram extends RegressionEvalMetric("signedpercentageerrorhistogram",
    "signed percentage error histogram", false)
}


/**
 * Forecast Metrics
 */
sealed abstract class ForecastEvalMetric
(
  val sparkEntryName: String,
  val humanFriendlyName: String,
  val isLargerBetter: Boolean
) extends EvalMetric


object ForecastEvalMetrics extends Enum[ForecastEvalMetric] {
  val values: Seq[ForecastEvalMetric] = findValues
  case object SMAPE extends ForecastEvalMetric("smape", "symmetric mean absolute percentage error", false)
  case object MASE extends ForecastEvalMetric("mase", "mean absolute scaled error", false)
  case object SeasonalError extends ForecastEvalMetric("seasonalError", "seasonal error", false)
}


/**
 * GeneralMetrics
 */
sealed abstract class OpEvaluatorNames
(
  val sparkEntryName: String,
  val humanFriendlyName: String,
  val isLargerBetter: Boolean // for default value
) extends EvalMetric

/**
 * Contains evaluator names used in logging
 */
object OpEvaluatorNames extends Enum[OpEvaluatorNames] {
  val values: Seq[OpEvaluatorNames] = findValues
  case object Binary extends OpEvaluatorNames("binEval", "binary evaluation metrics", true)
  case object BinScore extends OpEvaluatorNames("binScoreEval", "bin score evaluation metrics", false)
  case object Multi extends OpEvaluatorNames("multiEval", "multiclass evaluation metrics", true)
  case object Regression extends OpEvaluatorNames("regEval", "regression evaluation metrics", false)
  case object Forecast extends OpEvaluatorNames("regForecast", "regression evaluation metrics", false)
  case class Custom(name: String, humanName: String, largeBetter: Boolean) extends
    OpEvaluatorNames(name, humanName, largeBetter) {
    override def entryName: String = name.toLowerCase
  }
  def withName(name: String, isLargerBetter: Boolean): OpEvaluatorNames =
    super.withNameOption(name).getOrElse(Custom(name, name, isLargerBetter))

  def withNameInsensitive(name: String, isLargerBetter: Boolean): OpEvaluatorNames =
    super.withNameInsensitiveOption(name).getOrElse(Custom(name, name, isLargerBetter))

  def withFriendlyNameInsensitive(name: String): Option[OpEvaluatorNames] =
    values.collectFirst { case n if n.humanFriendlyName.equalsIgnoreCase(name) => n }
}
