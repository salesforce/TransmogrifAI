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

import com.fasterxml.jackson.databind.node.ObjectNode
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.utils.json.{JsonLike, JsonUtils}
import com.salesforce.op.utils.reflection.ReflectionUtils
import enumeratum.{Enum, EnumEntry}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}
import com.salesforce.op.utils.spark.RichMetadata._


/**
 * Trait for labelCol param
 */
trait OpHasLabelCol[T <: FeatureType] extends Params {
  final val labelCol: Param[String] = new Param[String](this, "labelCol", "label column name")
  setDefault(labelCol, "label")

  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setLabelCol(value: FeatureLike[T]): this.type = setLabelCol(value.name)
  def getLabelCol: String = $(labelCol)
}

/**
 * Trait for predictionCol param
 */
trait OpHasPredictionCol[T <: FeatureType] extends Params {
  final val predictionCol: Param[String] = new Param[String](this, "predictionCol", "prediction column name")
  setDefault(predictionCol, "prediction")

  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setPredictionCol(value: FeatureLike[T]): this.type = setPredictionCol(value.name)
  final def getPredictionCol: String = $(predictionCol)
}

/**
 * Trait for rawPredictionColParam
 */
trait OpHasRawPredictionCol[T <: FeatureType] extends Params {
  final val rawPredictionCol: Param[String] = new Param[String](
    this,
    "rawPredictionCol",
    "raw prediction (a.k.a. confidence) column name"
  )
  setDefault(rawPredictionCol, "rawPrediction")

  def setRawPredictionCol(value: String): this.type = set(rawPredictionCol, value)
  def setRawPredictionCol(value: FeatureLike[T]): this.type = setRawPredictionCol(value.name)
  final def getRawPredictionCol: String = $(rawPredictionCol)
}

/**
 * Trait for probabilityCol Param
 */
trait OpHasProbabilityCol[T <: FeatureType] extends Params {
  final val probabilityCol: Param[String] = new Param[String](
    this,
    "probabilityCol",
    "Column name for predicted class conditional probabilities." +
      " Note: Not all models output well-calibrated probability estimates!" +
      " These probabilities should be treated as confidences, not precise probabilities"
  )
  setDefault(probabilityCol, "probability")

  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)
  def setProbabilityCol(value: FeatureLike[T]): this.type = setProbabilityCol(value.name)
  final def getProbabilityCol: String = $(probabilityCol)
}


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
   * Convert metrics into metadata for saving
   * @return metadata
   */
  def toMetadata: Metadata = this.toMap.toMetadata
}


/**
 * Base Interface for OpEvaluator to be used in Evaluator creation. Can be used for both OP and spark
 * eval (so with workflows and cross validation).
 */
abstract class OpEvaluatorBase[T <: EvaluationMetrics] extends Evaluator {
  /**
   * Name of evaluator
   */
  val name: String = "Eval"

  /**
   * Evaluate function that returns a class or value with the calculated metric value(s).
   * @param dataset data to evaluate
   * @return metrics
   */
  def evaluateAll(dataset: Dataset[_]): T

  /**
   * Conversion from full metrics returned to a double value
   * @return double value used as spark eval number
   */
  def getDefaultMetric: T => Double

  final override def copy(extra: ParamMap): this.type = {
    val copy = ReflectionUtils.copy(this).asInstanceOf[this.type]
    copyValues(copy, extra)
  }

  /**
   * Evaluate function that returns a single metric
   * @param dataset data to evaluate
   * @return metric
   */
  override def evaluate(dataset: Dataset[_]): Double = getDefaultMetric(evaluateAll(dataset))
}

/**
 * Base Interface for OpClassificationEvaluator
 */
private[op] abstract class OpClassificationEvaluatorBase[T <: EvaluationMetrics]
  extends OpEvaluatorBase[T]
    with OpHasLabelCol[RealNN]
    with OpHasRawPredictionCol[OPVector]
    with OpHasProbabilityCol[OPVector]
    with OpHasPredictionCol[RealNN]

/**
 * Base Interface for OpBinaryClassificationEvaluator
 */
abstract class OpBinaryClassificationEvaluatorBase[T <: EvaluationMetrics]
(
  override val uid: String
) extends OpClassificationEvaluatorBase[T]


/**
 * Base Interface for OpMultiClassificationEvaluator
 */
abstract class OpMultiClassificationEvaluatorBase[T <: EvaluationMetrics]
(
  override val uid: String
) extends OpClassificationEvaluatorBase[T]


/**
 * Base Interface for OpRegressionEvaluator
 */
abstract class OpRegressionEvaluatorBase[T <: EvaluationMetrics]
(
  override val uid: String
) extends OpEvaluatorBase[T]
  with OpHasLabelCol[RealNN]
  with OpHasPredictionCol[RealNN]



sealed abstract class ClassificationEvalMetric(val sparkEntryName: String) extends EnumEntry with Serializable

/**
 * Binary Classification Metrics
 */
object BinaryClassEvalMetrics extends Enum[ClassificationEvalMetric] {
  val values = findValues
  case object Precision extends ClassificationEvalMetric("precision")
  case object Recall extends ClassificationEvalMetric("recall")
  case object F1 extends ClassificationEvalMetric("f1")
  case object Error extends ClassificationEvalMetric("accuracy")
  case object AuROC extends ClassificationEvalMetric("areaUnderROC")
  case object AuPR extends ClassificationEvalMetric("areaUnderPR")
}

/**
 * Multi Classification Metrics
 */
object MultiClassEvalMetrics extends Enum[ClassificationEvalMetric] {
  val values = findValues
  case object Precision extends ClassificationEvalMetric("weightedPrecision")
  case object Recall extends ClassificationEvalMetric("weightedRecall")
  case object F1 extends ClassificationEvalMetric("f1")
  case object Error extends ClassificationEvalMetric("accuracy")
  case object ThresholdMetrics extends ClassificationEvalMetric("thresholdMetrics")
}


/**
 * Contains the names of metrics used in logging
 */
private[op] case object OpMetricsNames {
  val rootMeanSquaredError = "root mean square error"
  val meanSquaredError = "mean square error"
  val meanAbsoluteError = "mean absolute error"
  val r2 = "r2"
  val auROC = "area under ROC"
  val auPR = "area under PR"
  val precision = "precision"
  val recall = "recall"
  val f1 = "f1"
  val accuracy = "accuracy"
  val error = "error"
  val tp = "true positive"
  val tn = "true negative"
  val fp = "false positive"
  val fn = "false negative"
}

/**
 * Contains evaluator names used in logging
 */
case object OpEvaluatorNames {
  val binary = "binEval"
  val multi = "multiEval"
  val regression = "regEval"
}


