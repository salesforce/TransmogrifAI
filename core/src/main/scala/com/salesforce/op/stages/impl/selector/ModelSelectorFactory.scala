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

import com.salesforce.op.evaluators.{EvaluationMetrics, OpEvaluatorBase}
import com.salesforce.op.stages.impl.ModelsToTry
import com.salesforce.op.stages.impl.selector.ModelSelectorNames.{EstimatorType, ModelType}
import com.salesforce.op.stages.impl.tuning.{OpValidator, Splitter}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

/**
 * Creates the model selector class
 */
trait ModelSelectorFactory {

  /**
   * Default model types and model parameters for problem type
   */
  trait ModelDefaults[M <: ModelsToTry] {

    /**
     * Subset of models to use in model selector
     */
    val modelTypesToUse: Seq[M]

    /**
     * Default models and parameters (must be a def) to use in model selector
     *
     * @return defaults for problem type
     */
    def modelsAndParams: Seq[(EstimatorType, ParamGridBuilder)]

  }

  /**
   * Create the model selector for specified interface
   *
   * @param validator           training split of cross validator
   * @param splitter            data prep class
   * @param trainTestEvaluators evaluation to do on data
   * @param modelTypesToUse     list of models to use
   * @param modelsAndParameters sequence of models and parameters to explore
   * @param modelDefaults       default model types and model parameters for problem type
   * @return model selector with these settings
   */
  protected def selector(
    validator: OpValidator[ModelType, EstimatorType],
    splitter: Option[Splitter],
    trainTestEvaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]],
    modelTypesToUse: Seq[ModelsToTry],
    modelsAndParameters: Seq[(EstimatorType, Array[ParamMap])],
    modelDefaults: ModelDefaults[_ <: ModelsToTry]
  ): ModelSelector[ModelType, EstimatorType] = {
    val modelTypeNames = modelTypesToUse.map(_.entryName).toSet
    val modelsToUse = {
      // if no models are specified use the defaults and filter by the named models to use
      if (modelsAndParameters.isEmpty) {
        modelDefaults.modelsAndParams
          .collect { case (e, grid) if modelTypeNames(e.getClass.getSimpleName) => e -> grid.build() }
      }
      // if models to use has been specified and the models have been specified - filter the models by the names
      else if (modelTypesToUse.toSet != modelDefaults.modelTypesToUse.toSet) {
        modelsAndParameters.filter { case (e, p) => modelTypeNames(e.getClass.getSimpleName) }
      }
      // else just use the specified models
      else modelsAndParameters
    }

    new ModelSelector(
      validator = validator,
      splitter = splitter,
      models = modelsToUse,
      evaluators = trainTestEvaluators
    )
  }

  protected def addDefaultEvaluators(
    evaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]],
    defaultEvaluators: Seq[OpEvaluatorBase[_ <: EvaluationMetrics]]
  ): Seq[OpEvaluatorBase[_ <: EvaluationMetrics]] = {
    defaultEvaluators.filter {
      e => !evaluators.exists(_.isInstanceOf[e.type])
    } ++ evaluators
  }

}
