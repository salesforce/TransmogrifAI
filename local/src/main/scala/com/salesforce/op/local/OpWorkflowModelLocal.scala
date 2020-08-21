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

package com.salesforce.op.local


import com.salesforce.op.OpWorkflowModel
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.stages.{OPStage, OpTransformer}

import scala.collection.mutable

/**
 * Enrichment for [[OpWorkflowModel]] to allow local scoring functionality
 */
trait OpWorkflowModelLocal extends Serializable {

  /**
   * Enrichment for [[OpWorkflowModel]] to allow local scoring functionality
   *
   * @param model [[OpWorkflowModel]]
   */
  implicit class RichOpWorkflowModel(model: OpWorkflowModel) {

    /**
     * Internal OP model representation
     *
     * @param output output name
     * @param model  model instance
     */
    private case class OPModel(output: String, model: OPStage with OpTransformer)

    /**
     * Internal MLeap model representation
     *
     * @param inputs  model inputs
     * @param output  model output
     * @param modelFn model function
     */
    private case class MLeapModel
    (
      inputs: Array[String],
      output: String,
      modelFn: Array[Any] => Any
    )

    /**
     * Prepares a score function for local scoring
     *
     * @return score function for local scoring
     */
    def scoreFunction: ScoreFunction = {
      // Prepare the stages for scoring
      val stagesWithIndex = model.getStages().zipWithIndex

      // Collect all OP stages
      // todo fix wrapped predictors
      val opStages = stagesWithIndex.collect { case (s: OpTransformer, i) => OPModel(s.getOutputFeatureName, s) -> i }

      // Collect all Spark wrapped stages
      val mleapStages = stagesWithIndex.filterNot(_._1.isInstanceOf[OpTransformer]).collect {
        case (opStage: OPStage with SparkWrapperParams[_], i) if opStage.getLocalMlStage().isDefined =>
          val model = opStage.getLocalMlStage().get
          MLeapModel(
            inputs = opStage.getTransientFeatures().map(_.name),
            output = opStage.getOutputFeatureName,
            modelFn = MLeapModelConverter.modelToFunction(model)
          ) -> i
      }

      // Combine all stages and apply the original order
      val allStages = (opStages ++ mleapStages).sortBy(_._2).map(_._1)
      val resultFeatures = model.getResultFeatures().map(_.name).toSet

      // Score Function
      input: Map[String, Any] => {
        val inputMap = mutable.Map.empty ++= input
        val transformedRow = allStages.foldLeft(inputMap) {
          // For OP Models we simply call transform
          case (row, OPModel(output, stage)) =>
            row += output -> stage.transformKeyValue(row.apply)

          // For MLeap models we call the prepared local model
          case (row, MLeapModel(inputs, output, modelFn)) =>
            val in = inputs.map(inputName => row.get(inputName) match {
              case None | Some(null) => null
              case Some(v) => v
            })
            row += output -> modelFn(in)
        }

        // Only return the result features of the model
        transformedRow.filterKeys(resultFeatures.contains).toMap
      }
    }

  }

}
