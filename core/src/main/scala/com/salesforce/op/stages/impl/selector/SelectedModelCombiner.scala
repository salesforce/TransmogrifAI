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
import com.salesforce.op.stages.base.ternary.{TernaryEstimator, TernaryModel}
import enumeratum.{Enum, EnumEntry}
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.Dataset



trait SelectedModelCombinerParams extends Params {

  final val combinationStrategy = new Param[String](parent = this, name = "combinationStrategy",
    doc = "Method used to combine predictions",
    isValid = (in: String) => CombinationStrategy.values.contains(in)
  )

}

class SelectedModelCombiner(uid: String = UID[SelectedModelCombiner]) extends
  TernaryEstimator[RealNN, Prediction, Prediction, Prediction]( operationName = "combineModels",
    uid = uid
  ) with SelectedModelCombinerParams with HasEval {

  private var p1weight = 0.0
  private var p2weight = 0.0
  private var metricName = _

  override def evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = {
    val ev1 = in1.getFeature().originStage.asInstanceOf[ModelSelector].evaluators
    val ev1names = ev1.map(_.name).toSet
    val ev2 = in2.getFeature().originStage.asInstanceOf[ModelSelector].evaluators
    ev1 ++ ev2.filter(e => ev1names.contains(e.name))
  }

  override protected[op] def outputsColNamesMap: Map[String, String] =
    Map(ModelSelectorNames.outputParamName -> getOutputFeatureName)

  override protected[op] def labelColName: String = in1.name

  private def getSummary(feature: TransientFeature) =
    ModelSelectorSummary.fromMetadata(getInputSchema()(feature.name).metadata)

  override def onSetInput() = {
    super.onSetInput()

    require(
      in1.getFeature().originStage.isInstanceOf[ModelSelector[_, _]] &&
        in2.getFeature().originStage.isInstanceOf[ModelSelector[_, _]],
      "Predictions must be from model selectors - other types of model are not supported at this time"
    )

    val summary1 = getSummary(in1)
    val summary2 = getSummary(in2)

    require(summary1.problemType == summary2.problemType,
      s"Cannot combine model selectors for different problem types found ${summary1.problemType}" +
        s" and ${summary2.problemType}")

    val eval1 = summary1.evaluationMetric
    val eval2 = summary2.evaluationMetric

    def getMetricValue(metrics: EvaluationMetrics, name: EvalMetric) = metrics.toMap.get(name.entryName)
      .map(_.asInstanceOf[Double])

    def getWinningModelMetric(summary: ModelSelectorSummary) =
      summary.validationResults.collectFirst{
        case r if r.modelUID == summary.bestModelUID => getMetricValue(r.metricValues, summary.evaluationMetric)
      }.flatten

    val (metricValueOpt1, metricValueOpt2) =
      if (eval1 == eval2) { // same decision metric in validation results
        metricName = eval1
        (getWinningModelMetric(summary1), getWinningModelMetric(summary2))
      } else { // look for overlapping metrics in training results
        val m2e1 = getMetricValue(summary2.trainEvaluation, eval1)
        val m1e2 = getMetricValue(summary1.trainEvaluation, eval2)
        if (m2e1.nonEmpty) {
          metricName = eval1
          (getMetricValue(summary1.trainEvaluation, eval1), m2e1)
        } else if (m1e2.nonEmpty) {
          metricName = eval2
          (m1e2, getMetricValue(summary2.trainEvaluation, eval2))
        } else (None, None)
      }

    def getMet(optionMet: Option[Double]) = optionMet.getOrElse {
      throw new RuntimeException("Evaluation metrics for two model selectors are non-overlapping")
    }

    val (metricValue1, metricValue2) = (getMet(metricValueOpt1), getMet(metricValueOpt2))

    (CombinationStrategy.withName($(combinationStrategy)), metricValue1 > metricValue2) match {
      case (CombinationStrategy.Best, true) => setMetadata(summary1.toMetadata())
        p1weight = 1.0
      case (CombinationStrategy.Best, false) => setMetadata(summary2.toMetadata())
        p2weight = 1.0
      case (CombinationStrategy.Weighted, _) =>
        p1weight = metricValue1 / (metricValue1 + metricValue2)
        p2weight = metricValue2 / (metricValue1 + metricValue2)
    }
  }



  override def fitFn(
    dataset: Dataset[(Option[Double],
      Map[String, Double],
      Map[String, Double])]
  ): TernaryModel[RealNN, Prediction, Prediction, Prediction] = {
    def updateKeys(map: Map[String, Any], string: String) = map.map{ case (k, v) => k + string -> v }

    if (CombinationStrategy.withName($(combinationStrategy)) == CombinationStrategy.Weighted) {
      val summary1 = getSummary(in1)
      val summary2 = getSummary(in2)
      val summary = new ModelSelectorSummary(
        validationType = summary1.validationType,
        validationParameters = updateKeys(summary1.validationParameters, "_1") ++
          updateKeys(summary2.validationParameters, "_2"),
        dataPrepParameters = updateKeys(summary1.dataPrepParameters, "_1") ++
          updateKeys(summary2.dataPrepParameters, "_2"),
        dataPrepResults = summary1.dataPrepResults.orElse(summary2.dataPrepResults),
        evaluationMetric = metricName,
        problemType = summary1.problemType,
        bestModelUID = summary1.bestModelUID + summary2.bestModelUID,
        bestModelName = summary1.bestModelName + summary2.bestModelName,
        bestModelType = summary1.bestModelType + summary2.bestModelType,
        validationResults = summary1.validationResults ++ summary2.validationResults,
        trainEvaluation = evaluate(dataset),
        holdoutEvaluation = None
      )
      setMetadata(summary.toMetadata())
    }
    new SelectedModelCombinerModel(weight1 = p1weight, weight2 = p2weight, operationName = operationName, uid = uid)
      .setEvaluators(evaluators)
  }

}

final class SelectedModelCombinerModel(weight1: Double, weight2: Double, operationName: String, uid: String) extends
  TernaryModel[RealNN, Prediction, Prediction, Prediction](operationName = operationName, uid = uid) with HasTestEval {

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

sealed abstract class CombinationStrategy(val name: String) extends EnumEntry with Serializable

object CombinationStrategy extends Enum[CombinationStrategy] {
  val values = findValues
  case object Weighted extends CombinationStrategy("weighted")
  case object Best extends CombinationStrategy("best")
}
