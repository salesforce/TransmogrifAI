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

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.evaluators._
import com.salesforce.op.stages.impl.MetadataLike
import com.salesforce.op.stages.impl.selector.ModelSelectorSummary._
import com.salesforce.op.stages.impl.tuning.{OpCrossValidation, OpTrainValidationSplit, OpValidator, SplitterSummary}
import com.salesforce.op.utils.json.JsonUtils
import com.salesforce.op.utils.reflection.ReflectionUtils
import com.salesforce.op.utils.spark.RichMetadata._
import enumeratum._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

import scala.reflect.ClassTag
import scala.util.{Failure, Try}

/**
 * This is used to store all information about fitting and model selection generated by the model selector class
 * @param validationType type of validation performed to select hyper parameters
 * @param validationParameters parameters on validation
 * @param dataPrepParameters parameters on data preparation before hyper parameter tuning
 * @param dataPrepResults changes made to the data in data preparation
 * @param evaluationMetric metric used to select hyper parameters and model
 * @param problemType type of modeling (eg binary classification, regressionm etc)
 * @param bestModelUID best model UID
 * @param bestModelName: best model unique name
 * @param bestModelType: best model type
 * @param holdoutEvaluation winning model performance on holdout data set
 * @param trainEvaluation winning model performance on training data set
 * @param validationResults model with parameters and metric for all evaluated
 */
case class ModelSelectorSummary
(
  validationType: ValidationType,
  validationParameters: Map[String, Any],
  dataPrepParameters: Map[String, Any],
  dataPrepResults: Option[SplitterSummary],
  evaluationMetric: EvalMetric,
  problemType: ProblemType,
  bestModelUID: String,
  bestModelName: String,
  bestModelType: String,
  validationResults: Seq[ModelEvaluation],
  trainEvaluation: EvaluationMetrics,
  holdoutEvaluation: Option[EvaluationMetrics]
) extends MetadataLike {

  /**
   * Converts to [[Metadata]]
   *
   * @param skipUnsupported skip unsupported values
   * @throws RuntimeException in case of unsupported value type
   * @return [[Metadata]] metadata
   */
  def toMetadata(skipUnsupported: Boolean): Metadata = {
    val meta = new MetadataBuilder()
      .putString(ValidationTypeName, validationType.entryName)
      .putMetadata(ValidationParameters, validationParameters.toMetadata(skipUnsupported))
      .putMetadata(DataPrepParameters, dataPrepParameters.toMetadata(skipUnsupported))
      .putString(EvaluationMetric, evaluationMetric.entryName)
      .putString(ProblemTypeName, problemType.entryName)
      .putString(BestModelUID, bestModelUID)
      .putString(BestModelName, bestModelName)
      .putString(BestModelType, bestModelType)
      .putMetadataArray(ValidationResults, validationResults.map(_.toMetadata(skipUnsupported)).toArray)
      .putStringArray(TrainEvaluation,
        Array(trainEvaluation.getClass.getName, trainEvaluation.toJson(pretty = false)))

    dataPrepResults.map(dp => meta.putMetadata(DataPrepResults, dp.toMetadata(skipUnsupported)))
    holdoutEvaluation.map(he => meta.putStringArray(HoldoutEvaluation,
      Array(he.getClass.getName, he.toJson(pretty = false))))
    meta.build()
  }

}

/**
 * Evaluation summary of model
 * @param modelUID uid for winning model
 * @param modelName unique name for model run
 * @param modelType simple name of type of model
 * @param metricValues evaluation metrics for model
 * @param modelParameters parameter settings for model
 */
case class ModelEvaluation
(
  modelUID: String,
  modelName: String,
  modelType: String,
  metricValues: EvaluationMetrics,
  modelParameters: Map[String, Any]
) extends MetadataLike {

  /**
   * Converts to [[Metadata]]
   *
   * @param skipUnsupported skip unsupported values
   * @throws RuntimeException in case of unsupported value type
   * @return [[Metadata]] metadata
   */
  def toMetadata(skipUnsupported: Boolean): Metadata = {
    new MetadataBuilder()
      .putString(ModelUID, modelUID)
      .putString(ModelName, modelName)
      .putString(ModelTypeName, modelType)
      .putStringArray(MetricValues, Array(metricValues.getClass.getName, metricValues.toJson(pretty = false)))
      .putMetadata(ModelParameters, modelParameters.toMetadata(skipUnsupported))
      .build()
  }
}


case object ModelSelectorSummary {

  val ValidationTypeName: String = "ValidationType"
  val ValidationParameters: String = "ValidationParameters"
  val DataPrepParameters: String = "DataPrepParameters"
  val DataPrepResults: String = "DataPrepResults"
  val EvaluationMetric: String = "EvaluationMetric"
  val ProblemTypeName: String = "ProblemType"
  val BestModelUID: String = "BestModelUID"
  val BestModelName: String = "BestModelName"
  val BestModelType: String = "BestModelType"
  val ValidationResults: String = "ValidationResults"
  val TrainEvaluation: String = "TrainEvaluation"
  val HoldoutEvaluation: String = "HoldoutEvaluation"

  val ModelUID: String = "ModelUID"
  val ModelName: String = "ModelName"
  val ModelTypeName: String = "ModelType"
  val MetricValues: String = "MetricValues"
  val ModelParameters: String = "ModelParameters"

  /**
   * Create case class from the metadata stored version of this class
   * @param meta metadata for this class
   * @return ModelSelectorSummary
   */
  def fromMetadata(meta: Metadata): ModelSelectorSummary = {

    def modelEvalFromMetadata(meta: Metadata): ModelEvaluation = {
      val wrapped = meta.wrapped
      val modelUID: String = wrapped.get[String](ModelUID)
      val modelName: String = wrapped.get[String](ModelName)
      val modelType: String = wrapped.get[String](ModelTypeName)
      val Array(metName, metJson) = wrapped.get[Array[String]](MetricValues)
      val metricValues: EvaluationMetrics = evalMetFromJson(metName, metJson).get
      val modelParameters: Map[String, Any] = wrapped.get[Metadata](ModelParameters).wrapped.underlyingMap

      ModelEvaluation(
        modelUID = modelUID,
        modelName = modelName,
        modelType = modelType,
        metricValues = metricValues,
        modelParameters = modelParameters
      )
    }


    val wrapped = meta.wrapped

    val validationType: ValidationType = ValidationType.withName(wrapped.get[String](ValidationTypeName))
    val validationParameters: Map[String, Any] = wrapped.get[Metadata](ValidationParameters)
      .wrapped.underlyingMap
    val dataPrepParameters: Map[String, Any] = wrapped.get[Metadata](DataPrepParameters)
      .wrapped.underlyingMap
    val dataPrepResults: Option[SplitterSummary] =
      if (wrapped.contains(DataPrepResults)) {
        SplitterSummary.fromMetadata(wrapped.get[Metadata](DataPrepResults)).toOption
      } else None
    val evaluationMetric: EvalMetric = EvalMetric.withNameInsensitive(wrapped.get[String](EvaluationMetric))
    val problemType: ProblemType = ProblemType.withName(wrapped.get[String](ProblemTypeName))
    val bestModelUID: String = wrapped.get[String](BestModelUID)
    val bestModelName: String = wrapped.get[String](BestModelName)
    val bestModelType: String = wrapped.get[String](BestModelType)
    val validationResults: Seq[ModelEvaluation] = wrapped.get[Array[Metadata]](ValidationResults)
      .map(modelEvalFromMetadata)
    val Array(metName, metJson) = wrapped.get[Array[String]](TrainEvaluation)
    val holdoutEvaluation: Option[EvaluationMetrics] =
      if (wrapped.contains(HoldoutEvaluation)) {
        val Array(metNameHold, metJsonHold) = wrapped.get[Array[String]](HoldoutEvaluation)
        evalMetFromJson(metNameHold, metJsonHold).toOption
      } else None

    ModelSelectorSummary(
      validationType = validationType,
      validationParameters = validationParameters,
      dataPrepParameters = dataPrepParameters,
      dataPrepResults = dataPrepResults,
      evaluationMetric = evaluationMetric,
      problemType = problemType,
      bestModelUID = bestModelUID,
      bestModelName = bestModelName,
      bestModelType = bestModelType,
      validationResults = validationResults,
      trainEvaluation = evalMetFromJson(metName, metJson).get,
      holdoutEvaluation = holdoutEvaluation)

  }

  /**
   * Decode metric values from JSON string
   *
   * @param json encoded metrics
   */
  private[selector] def evalMetFromJson(className: String, json: String): Try[EvaluationMetrics] = {
    def error(c: Class[_], t: Throwable): Try[MultiMetrics] = Failure[MultiMetrics] {
      new IllegalArgumentException(s"Could not extract metrics of type $c from: $json", t)
    }

    ReflectionUtils.classForName(className) match {
      case n if n == classOf[MultiMetrics] =>
        JsonUtils.fromString[Map[String, Map[String, Any]]](json).map{ d =>
          val asMetrics = d.flatMap{ case (_, values) =>
            values.map{
            case (nm: String, mp: Map[String, Any]@unchecked) =>
              val valsJson = JsonUtils.toJsonString(mp) // TODO: gross but it works. try to find a better way

              val binary = classOf[BinaryClassificationMetrics].getDeclaredFields.map(f => f.getName).toSet
              val multi = classOf[MultiClassificationMetrics].getDeclaredFields.map(f => f.getName).toSet
              val binscore = classOf[BinaryClassificationBinMetrics].getDeclaredFields.map(f => f.getName).toSet
              val regression = classOf[RegressionMetrics].getDeclaredFields.map(f => f.getName).toSet
              mp.keys match {
                case `binary` =>
                  nm -> JsonUtils.fromString[BinaryClassificationMetrics](valsJson).get
                case `binscore` =>
                  nm -> JsonUtils.fromString[BinaryClassificationBinMetrics](valsJson).get
                case `multi` =>
                  nm -> JsonUtils.fromString[MultiClassificationMetrics](valsJson).get
                case `regression` =>
                  nm -> JsonUtils.fromString[RegressionMetrics](valsJson).get
                case _ =>
                  nm -> JsonUtils.fromString[SingleMetric](valsJson).get
              }}
          }
          MultiMetrics(asMetrics)
        }.recoverWith { case t: Throwable => error(n, t) }
      case n => JsonUtils.fromString(json)(ClassTag(n))
        .recoverWith { case t: Throwable => error(n, t) }
    }
  }
}


sealed trait ProblemType extends EnumEntry with Serializable

object ProblemType extends Enum[ProblemType] {
  val values = findValues
  case object BinaryClassification extends ProblemType
  case object MultiClassification extends ProblemType
  case object Regression extends ProblemType
  case object Unknown extends ProblemType

  def fromEvalMetrics(eval: EvaluationMetrics): ProblemType = {
    eval match {
      case _: BinaryClassificationMetrics => ProblemType.BinaryClassification
      case _: BinaryClassificationBinMetrics => ProblemType.BinaryClassification
      case _: MultiClassificationMetrics => ProblemType.MultiClassification
      case _: RegressionMetrics => ProblemType.Regression
      case m: MultiMetrics =>
        val keys = m.metrics.keySet
        if (keys.exists(_.contains(OpEvaluatorNames.Binary.humanFriendlyName))) ProblemType.BinaryClassification
        else if (keys.exists(_.contains(OpEvaluatorNames.BinScore.humanFriendlyName))) ProblemType.BinaryClassification
        else if (keys.exists(_.contains(OpEvaluatorNames.Multi.humanFriendlyName))) ProblemType.MultiClassification
        else if (keys.exists(_.contains(OpEvaluatorNames.Regression.humanFriendlyName))) ProblemType.Regression
        else ProblemType.Unknown
      case _ => ProblemType.Unknown
    }
  }
}

sealed abstract class ValidationType(val humanFriendlyName: String) extends EnumEntry with Serializable
object ValidationType extends Enum[ValidationType] {
  val values = findValues
  case object CrossValidation extends ValidationType("Cross Validation")
  case object TrainValidationSplit extends ValidationType("Train Validation Split")

  def fromValidator(validator: OpValidator[_, _]): ValidationType = {
    validator match {
      case _: OpCrossValidation[_, _] => CrossValidation
      case _: OpTrainValidationSplit[_, _] => TrainValidationSplit
      case _ => throw new IllegalArgumentException(s"Unknown validator type $validator")
    }
  }
}


