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

import com.ibm.aardpfark.spark.ml.SparkSupport
import com.opendatagroup.hadrian.jvmcompiler.PFAEngine
import com.salesforce.op.OpWorkflowModel
import com.salesforce.op.features.types.OPVector
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.stages.{OPStage, OpPipelineStage, OpTransformer}
import com.salesforce.op.utils.json.JsonUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.Vector

/**
 * [[OpWorkflowModel]] enrichment functionality for local scoring
 */
trait OpWorkflowModelLocal {

  /**
   * [[OpWorkflowModel]] enrichment functionality for local scoring
   *
   * @param model [[OpWorkflowModel]]
   */
  implicit class OpWorkflowModelLocal(model: OpWorkflowModel) {

    /**
     * Prepares a score function for local scoring
     *
     * @return score function for local scoring
     */
    def scoreFunction: ScoreFunction = {
      val resultFeatures = model.getResultFeatures().map(_.name).toSet
      val stagesWithIndex = model.stages.zipWithIndex
      val opStages = stagesWithIndex.collect { case (s: OpTransformer, i) => s -> i }
      val sparkStages = stagesWithIndex.filterNot(_._1.isInstanceOf[OpTransformer]).collect {
        case (s: OPStage with SparkWrapperParams[_], i) if s.getSparkMlStage().isDefined =>
          ((s, s.getSparkMlStage().get.asInstanceOf[Transformer]), i)
      }
      val pfaEngines = for {
        ((s, sparkStage), i) <- sparkStages
        pfaJson = SparkSupport.toPFA(sparkStage, pretty = true)
        pfaEngine = PFAEngine.fromJson(pfaJson).head
      } yield ((s, pfaEngine), i)

      val allStages = (opStages ++ pfaEngines).sortBy(_._2)

      row => {
        val rowMap = collection.mutable.Map.empty ++ row
        val transformedRow = allStages.foldLeft(rowMap) {
          case (r, (s: OPStage with OpTransformer, i)) =>
            r += s.getOutputFeatureName -> s.transformKeyValue(r.apply)

          case (r, (engine: PFAEngine[AnyRef, AnyRef]@unchecked, i)) =>
            val stage = stagesWithIndex.find(_._2 == i).map(_._1.asInstanceOf[OpPipelineStage[_]]).get
            val outName = stage.getOutputFeatureName
            val inputName = stage.getInputFeatures().collect {
              case f if f.isSubtypeOf[OPVector] => f.name
            }.head
            val vector = r(inputName).asInstanceOf[Vector].toArray
            val input = s"""{"$inputName":${vector.mkString("[", ",", "]")}}"""
            val engineIn = engine.jsonInput(input)
            val res = engine.action(engineIn).toString
            r += outName -> JsonUtils.fromString[Map[String, Any]](res).get
        }
        transformedRow.filterKeys(resultFeatures.contains).toMap
      }
    }
  }

}
