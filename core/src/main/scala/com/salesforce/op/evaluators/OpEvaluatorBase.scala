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
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.sql.Dataset


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
 * Base Interface for OpEvaluator to be used in Evaluator creation. Can be used for both OP and spark
 * eval (so with workflows and cross validation).
 */
abstract class OpEvaluatorBase[T <: EvaluationMetrics] extends Evaluator
  with OpHasLabelCol[RealNN] with OpHasPredictionValueCol[RealNN] with OpHasPredictionCol {

  /**
   * Name of evaluator
   */
  val name: EvalMetric


  /**
   * Use the definition of the metric to determine if larger is better
   * @return
   */
  override def isLargerBetter: Boolean = name.isLargerBetter

  /**
   * Evaluate function that returns a class or value with the calculated metric value(s).
 *
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
  extends OpEvaluatorBase[T] with OpHasRawPredictionCol[OPVector] with OpHasProbabilityCol[OPVector] {

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
abstract class OpBinaryClassificationEvaluatorBase[T <: EvaluationMetrics](override val uid: String)
  extends OpClassificationEvaluatorBase[T]


/**
 * Base Interface for OpMultiClassificationEvaluator
 */
abstract class OpMultiClassificationEvaluatorBase[T <: EvaluationMetrics](override val uid: String)
  extends OpClassificationEvaluatorBase[T]


/**
 * Base Interface for OpRegressionEvaluator
 */
abstract class OpRegressionEvaluatorBase[T <: EvaluationMetrics](override val uid: String)
  extends OpEvaluatorBase[T] {

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
