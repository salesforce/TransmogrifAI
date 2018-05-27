/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.evaluators.{OpBinaryClassificationEvaluatorBase, OpEvaluatorBase, OpMultiClassificationEvaluatorBase}
import com.salesforce.op.stages.impl.selector.ModelInfo
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.MetadataBuilder
import org.slf4j.{Logger, LoggerFactory}

/**
 * Best Model container
 *
 * @param name     the name of the best model
 * @param model    best trained model
 * @param metadata optional metadata
 * @tparam M model type
 */
case class BestModel[M <: Model[_]](name: String, model: M, metadata: Option[MetadataBuilder] = None)

/**
 * Best Estimator container
 *
 * @param name     the name of the best model
 * @param estimator    best estimator
 * @param metadata optional metadata
 * @tparam E model type
 */
case class BestEstimator[E <: Estimator[_]](name: String, estimator: E, metadata: MetadataBuilder = new MetadataBuilder)

/**
 * Validated Model container
 *
 * @param model     model instance
 * @param bestIndex best metric / grid index
 * @param metrics   all computed metrics
 * @param grids     all param grids
 */
private[tuning] case class ValidatedModel[E <: Estimator[_]]
(
  model: E,
  bestIndex: Int,
  metrics: Array[Double],
  grids: Array[ParamMap]
) {
  /**
   * Best metric (metric at bestIndex)
   */
  def bestMetric: Double = metrics(bestIndex)
  /**
   * Best grid (param grid at bestIndex)
   */
  def bestGrid: ParamMap = grids(bestIndex)
}

/**
 * Abstract class for Validator: Cross Validation or Train Validation Split
 * The model type should be the output of the estimator type but specifying that kills the scala compiler
 */
private[impl] trait OpValidator[M <: Model[_], E <: Estimator[_]] extends Serializable {

  @transient protected lazy val log: Logger = LoggerFactory.getLogger(this.getClass)

  def seed: Long
  def evaluator: OpEvaluatorBase[_]
  def validationName: String
  def stratify: Boolean

  private[op] final def isClassification = evaluator match {
    case _: OpBinaryClassificationEvaluatorBase[_] => true
    case _: OpMultiClassificationEvaluatorBase[_] => true
    case _ => false
  }

  /**
   * Function that performs the model selection
   *
   * @param modelInfo estimators and grids to validate
   * @param label
   * @param features
   * @param dataset
   * @return estimator
   */
  private[op] def validate(
    modelInfo: Seq[ModelInfo[E]],
    dataset: Dataset[_],
    label: String,
    features: String
  ): BestModel[M]

  /**
   * Get the best model and the metadata with with the validator params
   *
   * @param modelsFit info from validation
   * @param bestModel best fit model
   * @param splitInfo split info for logging
   * @return best model
   */
  private[op] def wrapBestModel(
    modelsFit: Array[ValidatedModel[E]],
    bestModel: M,
    splitInfo: String
  ): BestModel[M] = {
    log.info(
      "Model Selection over {} with {} with {} and the {} metric",
      modelsFit.map(_.model.getClass.getSimpleName).mkString(","), validationName, splitInfo, evaluator.name
    )
    val meta = new MetadataBuilder()
    val cvFittedModels = modelsFit.map(v => updateBestModelMetadata(meta, v) -> v.bestMetric)
    val newMeta = new MetadataBuilder().putMetadata(validationName, meta.build())
    val (bestModelName, _) = if (evaluator.isLargerBetter) cvFittedModels.maxBy(_._2) else cvFittedModels.minBy(_._2)

    BestModel(name = bestModelName, model = bestModel, metadata = Option(newMeta))
  }

  /**
   * Update metadata during model selection and return best model name
   * @return best model name
   */
  private[op] def updateBestModelMetadata(metadataBuilder: MetadataBuilder, v: ValidatedModel[E]): String = {
    val ValidatedModel(model, bestIndex, metrics, grids) = v
    val modelParams = model.extractParamMap()
    def makeModelName(index: Int) = s"${model.uid}_$index"

    for {((paramGrid, met), ind) <- grids.zip(metrics).zipWithIndex} {
      val paramMetBuilder = new MetadataBuilder()
      paramMetBuilder.putString(evaluator.name, met.toString)
      // put in all model params from the winner
      modelParams.toSeq.foreach(p => paramMetBuilder.putString(p.param.name, p.value.toString))
      // override with param map values if they exists
      paramGrid.toSeq.foreach(p => paramMetBuilder.putString(p.param.name, p.value.toString))
      metadataBuilder.putMetadata(makeModelName(ind), paramMetBuilder.build())
    }
    makeModelName(bestIndex)
  }

  /**
   * Creates Train Validation Splits
   * @param rdd
   * @return Train Validation Splits
   */
  private[op] def createTrainValidationSplits(rdd: RDD[(Double, Vector, String)]): Array[(RDD[Row], RDD[Row])]
}

object ValidatorParamDefaults {
  def Seed: Long = util.Random.nextLong // scalastyle:off method.name
  val labelCol = "labelCol"
  val NumFolds = 3
  val TrainRatio = 0.75
  val Stratify = false
}

