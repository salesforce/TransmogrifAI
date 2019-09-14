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

import com.salesforce.op.UID
import com.salesforce.op.evaluators.{EvalMetric, EvaluationMetrics, OpEvaluatorBase}
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types.{Prediction, RealNN}
import com.salesforce.op.stages.OpPipelineStage3
import com.salesforce.op.stages.base.ternary.OpTransformer3
import com.salesforce.op.stages.impl.feature.CombinationStrategy
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe._

/**
 * Parameters for SelectorCombiner
 */
trait SelectedCombinerParams extends Params {

  final val combinationStrategy = new Param[String](parent = this, name = "combinationStrategy",
    doc = "Method used to combine predictions",
    isValid = (in: String) => CombinationStrategy.values.map(_.entryName).contains(in)
  )
  def setCombinationStrategy(value: CombinationStrategy): this.type = set(combinationStrategy, value.entryName)
  def getCombinationStrategy(): CombinationStrategy = CombinationStrategy.withNameInsensitive($(combinationStrategy))
  setDefault(combinationStrategy, CombinationStrategy.Best.entryName)

}

/**
 * Class used to combine the predictions produced by two model selectors into a single prediction.
 * Does this by either taking the best models prediction or a combination of the two predictions that is
 * either weighted by the accuracy measure or equal. Uses the summary information from the model selectors to
 * determine the accuracy of the predictions and reruns evaluation (both train and test) when the predictions
 * are combined.
 *
 * @param operationName name of operation
 * @param uid stage uid
 */
class SelectedModelCombiner
(
  val operationName: String = "combineModels",
  val uid: String = UID[SelectedModelCombiner]
)(
  implicit val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends Estimator[SelectedCombinerModel] with
  OpPipelineStage3[RealNN, Prediction, Prediction, Prediction] with SelectedCombinerParams with HasEval {

  override def evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = {
    val ms1 = in2.getFeature().originStage.asInstanceOf[ModelSelector[_, _]]
    val ev1 = ms1.evaluators
    val ev1names = ms1.evaluators.map(_.name).toSet
    val ms2 = in3.getFeature().originStage.asInstanceOf[ModelSelector[_, _]]
    val ev2 = ms2.evaluators
    ev1 ++ ev2.filterNot(e => ev1names.contains(e.name))
  }

  override protected[op] def outputsColNamesMap: Map[String, String] =
    Map(ModelSelectorNames.outputParamName -> getOutputFeatureName)

  override protected[op] def labelColName: String = in1.name

  private def getSummary(feature: TransientFeature) =
    ModelSelectorSummary.fromMetadata(getInputSchema()(feature.name).metadata.getSummaryMetadata())

  override def onSetInput(): Unit = {
    super.onSetInput()

    require(
      in2.getFeature().originStage.isInstanceOf[ModelSelector[_, _]] &&
        in3.getFeature().originStage.isInstanceOf[ModelSelector[_, _]],
      "Predictions must be from model selectors - other types of model are not supported at this time"
    )

  }

  override def fit(dataset: Dataset[_]): SelectedCombinerModel = {
    setInputSchema(dataset.schema).transformSchema(dataset.schema)

    val summary1 = getSummary(in2)
    val summary2 = getSummary(in3)

    require(summary1.problemType == summary2.problemType,
      s"Cannot combine model selectors for different problem types found ${summary1.problemType}" +
        s" and ${summary2.problemType}")

    val eval1 = summary1.evaluationMetric
    val eval2 = summary2.evaluationMetric

    val (metricValueOpt1, metricValueOpt2, metricName) =
      if (eval1 == eval2) { // same decision metric in validation results
        (getWinningModelMetric(summary1), getWinningModelMetric(summary2), eval1)
      } else { // look for overlapping metrics in training results
        val m2e1 = getMetricValue(summary2.trainEvaluation, eval1)
        val m1e2 = getMetricValue(summary1.trainEvaluation, eval2)
        if (m2e1.nonEmpty) {
          (getMetricValue(summary1.trainEvaluation, eval1), m2e1, eval1)
        } else if (m1e2.nonEmpty) {
          (m1e2, getMetricValue(summary2.trainEvaluation, eval2), eval2)
        } else (None, None, eval1)
      }

    val (metricValue1, metricValue2) = (getMet(metricValueOpt1), getMet(metricValueOpt2))

    val strategy = getCombinationStrategy()
    val (weight1, weight2) = strategy match {
      case CombinationStrategy.Best =>
        (metricValue1 > metricValue2, metricName.isLargerBetter) match {
          case (true, true) => (1.0, 0.0)
          case (true, false) => (0.0, 1.0)
          case (false, true) => (0.0, 1.0)
          case (false, false) => (1.0, 0.0)
        }
      case CombinationStrategy.Weighted =>
        (metricValue1 / (metricValue1 + metricValue2), metricValue2 / (metricValue1 + metricValue2))
      case CombinationStrategy.Equal =>
        (0.5, 0.5)
      case s => throw new RuntimeException(s"Combination strategy $s is not supported")
    }

    val model: SelectedCombinerModel = new SelectedCombinerModel(
      weight1 = weight1, weight2 = weight2, strategy = strategy, metric = metricName,
      operationName = operationName, uid = uid
    )
      .setEvaluators(evaluators)
      .setParent(this)
      .setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[Prediction], in3.asFeatureLike[Prediction])
      .setOutputFeatureName(getOutputFeatureName)

    if (model.strategy == CombinationStrategy.Best && model.weight1 > 0.5) {
      setMetadata(summary1.toMetadata().toSummaryMetadata())
    } else if (model.strategy == CombinationStrategy.Best) {
      setMetadata(summary2.toMetadata().toSummaryMetadata())
    } else {
      val summary = new ModelSelectorSummary(
        validationType = summary1.validationType,
        validationParameters = updateKeys(summary1.validationParameters, "_1") ++
          updateKeys(summary2.validationParameters, "_2"),
        dataPrepParameters = updateKeys(summary1.dataPrepParameters, "_1") ++
          updateKeys(summary2.dataPrepParameters, "_2"),
        dataPrepResults = summary1.dataPrepResults.orElse(summary2.dataPrepResults),
        evaluationMetric = metricName,
        problemType = summary1.problemType,
        bestModelUID = summary1.bestModelUID + " " + summary2.bestModelUID,
        bestModelName = summary1.bestModelName + " " + summary2.bestModelName,
        bestModelType = summary1.bestModelType + " " + summary2.bestModelType,
        validationResults = summary1.validationResults ++ summary2.validationResults,
        trainEvaluation = evaluate(model.transform(dataset)),
        holdoutEvaluation = None
      )
      setMetadata(summary.toMetadata().toSummaryMetadata())
    }

    model.setMetadata(getMetadata())
  }

  private def getMetricValue(metrics: EvaluationMetrics, name: EvalMetric) =
    metrics.toMap.collectFirst{
      case (k, v: Double) if k.contains(name.humanFriendlyName) || k.contains(name.entryName) => v
    }

  private def getWinningModelMetric(summary: ModelSelectorSummary) = {
    summary.validationResults.collectFirst {
      case r if r.modelUID == summary.bestModelUID =>
        getMetricValue(r.metricValues, summary.evaluationMetric)
    }.flatten
  }

  private def getMet(optionMet: Option[Double]) = optionMet.getOrElse {
    throw new RuntimeException("Evaluation metrics for two model selectors are non-overlapping")
  }

  private def updateKeys(map: Map[String, Any], string: String) = map.map{ case (k, v) => k + string -> v }

}

final class SelectedCombinerModel private[op]
(
  val weight1: Double,
  val weight2: Double,
  val strategy: CombinationStrategy,
  val metric: EvalMetric,
  val operationName: String,
  val uid: String
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[Prediction],
  val tti3: TypeTag[Prediction],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends Model[SelectedCombinerModel] with OpTransformer3[RealNN, Prediction, Prediction, Prediction]
  with HasTestEval {

  override protected[op] def outputsColNamesMap: Map[String, String] =
    Map(ModelSelectorNames.outputParamName -> getOutputFeatureName)

  override def transformFn: (RealNN, Prediction, Prediction) => Prediction = (_, p1: Prediction, p2: Prediction) => {
    val rawPrediction = p1.rawPrediction.zip(p2.rawPrediction).map{ case (v1, v2) => v1 * weight1 + v2 * weight2 }
    val probability = p1.probability.zip(p2.probability).map{ case (v1, v2) => v1 * weight1 + v2 * weight2 }
    val prediction =
      if (probability.nonEmpty) probability.indexOf(probability.max).toDouble
      else p1.prediction * weight1 + p2.prediction * weight2
    Prediction(prediction = prediction, probability = probability, rawPrediction = rawPrediction)
  }

  lazy val labelColName: String = in1.name

  @transient private var evaluatorList: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = Seq.empty
  def setEvaluators(ev: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]]): this.type = {
    evaluatorList = ev
    this
  }
  override def evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = evaluatorList
}

