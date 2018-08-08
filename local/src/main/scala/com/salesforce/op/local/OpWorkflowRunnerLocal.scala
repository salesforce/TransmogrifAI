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

import com.ibm.aardpfark.spark.ml.SparkSupport.toPFA
import com.opendatagroup.hadrian.jvmcompiler.PFAEngine
import com.salesforce.op.features.types.OPVector
import com.salesforce.op.stages.{OpPipelineStage, OpTransformer}
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.utils.json.JsonUtils
import com.salesforce.op.{OpParams, OpWorkflow}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.Vector


/**
 * A class for running an Optimus Prime Workflow without Spark.
 *
 * @param workflow the workflow that you want to run (Note: the workflow should have the resultFeatures set)
 */
class OpWorkflowRunnerLocal(val workflow: OpWorkflow) {

  type ScoreFun = Map[String, Any] => Map[String, Any]

  /**
   * Load the model & prepare a score local function
   *
   * @param params params to use during scoring
   * @return score local function
   */
  def score(params: OpParams): ScoreFun = {
    require(params.modelLocation.isDefined, "Model location must be set in params")
    val model = workflow.loadModel(params.modelLocation.get)

    val stagesWithIndex = model.stages.zipWithIndex
    val opStages = stagesWithIndex.collect { case (s: OpTransformer, i) => s -> i }
    val sparkStages = stagesWithIndex.collect {
      case (s: SparkWrapperParams[_], i) => s.getSparkMlStage().map(_ -> i)
      case (s: Transformer, i) if !s.isInstanceOf[OpTransformer] => Some(s -> i)
    }.flatten.map(v => v._1.asInstanceOf[Transformer] -> v._2)

    val pfaStages = sparkStages.map { case (s, i) => toPFA(s, pretty = true) -> i }
    val engines = pfaStages.map { case (s, i) => PFAEngine.fromJson(s, multiplicity = 1).head -> i }
    val loadedStages = (opStages ++ engines).sortBy(_._2)

    row: Map[String, Any] => {
      val rowMap = collection.mutable.Map.empty ++ row
      val transformedRow = loadedStages.foldLeft(rowMap) { (r, s) =>
        s match {
          case (s: OpTransformer, _) => {
            r += s.asInstanceOf[OpPipelineStage[_]].getOutputFeatureName -> s.transformKeyValue(r.apply)
          }
          case (e: PFAEngine[AnyRef, AnyRef], i) => {
            val stage = stagesWithIndex.find(_._2 == i).map(_._1.asInstanceOf[OpPipelineStage[_]]).get
            val outName = stage.getOutputFeatureName
            val inputName = stage.getInputFeatures().collect {
              case f if f.isSubtypeOf[OPVector] => f.name
            }.head
            val vector = r(inputName).asInstanceOf[Vector].toArray
            val input = s"""{"$inputName":${vector.mkString("[", ",", "]")}}"""
            val res = e.action(e.jsonInput(input)).toString
            r += outName -> JsonUtils.fromString[Map[String, Any]](res).get
          }
        }
      }
      val resultFeatures = model.getResultFeatures().map(_.name)
      transformedRow.collect { case r@(k, _) if resultFeatures.contains(k) => r }.toMap
    }
  }

}
