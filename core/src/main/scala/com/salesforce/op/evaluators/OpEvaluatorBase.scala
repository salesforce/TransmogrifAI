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

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.utils.json.{JsonLike, JsonUtils}
import com.salesforce.op.utils.reflection.ReflectionUtils
import com.salesforce.op.utils.spark.RichMetadata._
import enumeratum.{Enum, EnumEntry}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.Metadata

import scala.util.Try


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
 * Trait for predictionCol which contains all output results param
 */
trait OpHasPredictionCol extends Params {
  final val predictionCol: Param[String] = new Param[String](this, "predictionCol", "prediction column name")

  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setPredictionCol(value: FeatureLike[Prediction]): this.type = setPredictionCol(value.name)
  final def getPredictionCol: String = $(predictionCol)
}

/**
 * Trait for internal flattened predictionCol param
 */
trait OpHasPredictionValueCol[T <: FeatureType] extends Params {
  final val predictionValueCol: Param[String] = new Param[String](this, "predictionValueCol", "prediction column name")
  setDefault(predictionValueCol, "prediction")

  protected def setPredictionValueCol(value: String): this.type = set(predictionValueCol, value)
  protected def setPredictionValueCol(value: FeatureLike[T]): this.type = setPredictionValueCol(value.name)
  protected final def getPredictionValueCol: String = $(predictionValueCol)
}

/**
 * Trait for internal flattened rawPredictionColParam
 */
trait OpHasRawPredictionCol[T <: FeatureType] extends Params {
  final val rawPredictionCol: Param[String] = new Param[String](
    this, "rawPredictionCol", "raw prediction (a.k.a. confidence) column name"
  )
  setDefault(rawPredictionCol, "rawPrediction")

  protected def setRawPredictionCol(value: String): this.type = set(rawPredictionCol, value)
  protected def setRawPredictionCol(value: FeatureLike[T]): this.type = setRawPredictionCol(value.name)
  protected final def getRawPredictionCol: String = $(rawPredictionCol)
}

/**
 * Trait for internal flattened probabilityCol Param
 */
trait OpHasProbabilityCol[T <: FeatureType] extends Params {
  final val probabilityCol: Param[String] = new Param[String](
    this, "probabilityCol",
    "Column name for predicted class conditional probabilities." +
      " Note: Not all models output well-calibrated probability estimates!" +
      " These probabilities should be treated as confidences, not precise probabilities"
  )
  setDefault(probabilityCol, "probability")

  protected def setProbabilityCol(value: String): this.type = set(probabilityCol, value)
  protected def setProbabilityCol(value: FeatureLike[T]): this.type = setProbabilityCol(value.name)
  protected final def getProbabilityCol: String = $(probabilityCol)
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
abstract class OpEvaluatorBase[T <: EvaluationMetrics] extends Evaluator
  with OpHasLabelCol[RealNN]
  with OpHasPredictionValueCol[RealNN]
  with OpHasPredictionCol {
  /**
   * Name of evaluator
   */
  val name: EvalMetric

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

  /**
   * Prepare data with different input types so that it can be consumed by the evaluator
   * @param data input data
   * @param labelColName name of the label column
   * @return data formatted for use with the evaluator
   */
  protected def makeDataToUse(data: Dataset[_], labelColName: String): Dataset[_]
}

/**
 * Base Interface for OpClassificationEvaluator
 */
private[op] abstract class OpClassificationEvaluatorBase[T <: EvaluationMetrics]
  extends OpEvaluatorBase[T]
    with OpHasRawPredictionCol[OPVector]
    with OpHasProbabilityCol[OPVector] {

  /**
   * Prepare data with different input types so that it can be consumed by the evaluator
   * @param data input data
   * @param labelColName name of the label column
   * @return data formatted for use with the evaluator
   */
  protected def makeDataToUse(data: Dataset[_], labelColName: String): Dataset[_] = {
    if (isSet(predictionCol) &&
      !(isSet(predictionValueCol) && data.columns.contains(getPredictionValueCol))) {
      val fullPredictionColName = getPredictionCol
      val (predictionColName, rawPredictionColName, probabilityColName) =
        (s"${fullPredictionColName}_pred", s"${fullPredictionColName}_raw", s"${fullPredictionColName}_prob")

      setPredictionValueCol(predictionColName)
      setRawPredictionCol(rawPredictionColName)
      setProbabilityCol(probabilityColName)

      val flattenedData = data.select(labelColName, getPredictionCol).rdd.map{ r =>
        val label = r.getDouble(0)
        val predMap: Prediction = r.getMap[String, Double](1).toMap.toPrediction
        val raw = Vectors.dense(predMap.rawPrediction)
        val prob = Vectors.dense(predMap.probability)
        val probUse = if (prob.size == 0) raw else prob // so can calculate threshold metrics for LinearSVC
        (label, predMap.prediction, raw, probUse)
      }

      data.sqlContext.createDataFrame(flattenedData)
        .toDF(labelColName, predictionColName, rawPredictionColName, probabilityColName)

    } else data
  }
}

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
) extends OpEvaluatorBase[T] {

  /**
   * Prepare data with different input types so that it can be consumed by the evaluator
   * @param data input data
   * @param labelColName name of the label column
   * @return data formatted for use with the evaluator
   */
  protected def makeDataToUse(data: Dataset[_], labelColName: String): Dataset[_] = {
    if (isSet(predictionCol) &&
      !(isSet(predictionValueCol) && data.columns.contains(getPredictionValueCol))) {
      val fullPredictionColName = getPredictionCol
      val predictionColName = s"${fullPredictionColName}_pred"
      setPredictionValueCol(predictionColName)

      val flattenedData = data.select(labelColName, fullPredictionColName).rdd
        .map(r => (r.getDouble(0), r.getMap[String, Double](1).toMap.toPrediction.prediction ))

      data.sqlContext.createDataFrame(flattenedData).toDF(labelColName, predictionColName)

    } else data
  }
}

/**
 * Eval metric
 */
trait EvalMetric extends EnumEntry with Serializable {
  /**
   * Spark metric name
   */
  def sparkEntryName: String

  /**
   * Human friendly metric name
   */
  def humanFriendlyName: String

}

/**
 * Eval metric companion object
 */
object EvalMetric {

  def withNameInsensitive(name: String): EvalMetric = {
    BinaryClassEvalMetrics.withNameInsensitiveOption(name)
      .orElse(MultiClassEvalMetrics.withNameInsensitiveOption(name))
      .orElse(RegressionEvalMetrics.withNameInsensitiveOption(name))
      .orElse(OpEvaluatorNames.withNameInsensitiveOption(name))
      .getOrElse(OpEvaluatorNames.Custom(name, name))
  }
}

/**
 * Classification Metrics
 */
sealed abstract class ClassificationEvalMetric
(
  val sparkEntryName: String,
  val humanFriendlyName: String
) extends EvalMetric

/**
 * Binary Classification Metrics
 */
object BinaryClassEvalMetrics extends Enum[ClassificationEvalMetric] {
  val values = findValues
  case object Precision extends ClassificationEvalMetric("weightedPrecision", "precision")
  case object Recall extends ClassificationEvalMetric("weightedRecall", "recall")
  case object F1 extends ClassificationEvalMetric("f1", "f1")
  case object Error extends ClassificationEvalMetric("accuracy", "error")
  case object AuROC extends ClassificationEvalMetric("areaUnderROC", "area under ROC")
  case object AuPR extends ClassificationEvalMetric("areaUnderPR", "area under precision-recall")
  case object TP extends ClassificationEvalMetric("TP", "true positive")
  case object TN extends ClassificationEvalMetric("TN", "true negative")
  case object FP extends ClassificationEvalMetric("FP", "false positive")
  case object FN extends ClassificationEvalMetric("FN", "false negative")
}

/**
 * Multi Classification Metrics
 */
object MultiClassEvalMetrics extends Enum[ClassificationEvalMetric] {
  val values = findValues
  case object Precision extends ClassificationEvalMetric("weightedPrecision", "precision")
  case object Recall extends ClassificationEvalMetric("weightedRecall", "recall")
  case object F1 extends ClassificationEvalMetric("f1", "f1")
  case object Error extends ClassificationEvalMetric("accuracy", "error")
  case object ThresholdMetrics extends ClassificationEvalMetric("thresholdMetrics", "threshold metrics")
}


/**
 * Regression Metrics
 */
sealed abstract class RegressionEvalMetric
(
  val sparkEntryName: String,
  val humanFriendlyName: String
) extends EvalMetric

/**
 * Regression Metrics
 */
object RegressionEvalMetrics extends Enum[RegressionEvalMetric] {
  val values: Seq[RegressionEvalMetric] = findValues
  case object RootMeanSquaredError extends RegressionEvalMetric("rmse", "root mean square error")
  case object MeanSquaredError extends RegressionEvalMetric("mse", "mean square error")
  case object R2 extends RegressionEvalMetric("r2", "r2")
  case object MeanAbsoluteError extends RegressionEvalMetric("mae", "mean absolute error")
}


/**
 * GeneralMetrics
 */
sealed abstract class OpEvaluatorNames
(
  val sparkEntryName: String,
  val humanFriendlyName: String
) extends EvalMetric

/**
 * Contains evaluator names used in logging
 */
object OpEvaluatorNames extends Enum[OpEvaluatorNames] {
  val values: Seq[OpEvaluatorNames] = findValues
  case object Binary extends OpEvaluatorNames("binEval", "binary evaluation metics")
  case object Multi extends OpEvaluatorNames("multiEval", "multiclass evaluation metics")
  case object Regression extends OpEvaluatorNames("regEval", "regression evaluation metics")
  case class Custom(name: String, humanName: String) extends OpEvaluatorNames(name, humanName) {
    override def entryName: String = name.toLowerCase
  }
  override def withName(name: String): OpEvaluatorNames = Try(super.withName(name)).getOrElse(Custom(name, name))
  override def withNameInsensitive(name: String): OpEvaluatorNames = super.withNameInsensitiveOption(name)
    .getOrElse(Custom(name, name))
}
