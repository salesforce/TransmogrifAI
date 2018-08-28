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
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.stages.{OPStage, OpTransformer}
import org.apache.spark.ml.SparkMLSharedParamConstants._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization

import scala.collection.mutable

/**
 * Enrichment for [[OpWorkflowModel]] to allow local scoring functionality
 */
trait OpWorkflowModelLocal {

  /**
   * Enrichment for [[OpWorkflowModel]] to allow local scoring functionality
   *
   * @param model [[OpWorkflowModel]]
   */
  implicit class RichOpWorkflowModel(model: OpWorkflowModel) {

    private implicit val formats = DefaultFormats

    /**
     * Internal PFA model representation
     *
     * @param inputs mode inputs mappings
     * @param output output mapping
     * @param engine PFA engine
     */
    private case class PFAModel
    (
      inputs: Map[String, String],
      output: (String, String),
      engine: PFAEngine[AnyRef, AnyRef]
    )

    /**
     * Internal OP model representation
     *
     * @param output output name
     * @param model model instance
     */
    private case class OPModel(output: String, model: OPStage with OpTransformer)

    /**
     * Prepares a score function for local scoring
     *
     * @return score function for local scoring
     */
    def scoreFunction: ScoreFunction = {
      // Prepare the stages for scoring
      val stagesWithIndex = model.stages.zipWithIndex
      // Collect all OP stages
      val opStages = stagesWithIndex.collect { case (s: OpTransformer, i) => OPModel(s.getOutputFeatureName, s) -> i }
      // Collect all Spark wrapped stages
      val sparkStages = stagesWithIndex.filterNot(_._1.isInstanceOf[OpTransformer]).collect {
        case (s: OPStage with SparkWrapperParams[_], i) if s.getSparkMlStage().isDefined =>
          (s, s.getSparkMlStage().get.asInstanceOf[Transformer].copy(ParamMap.empty), i)
      }
      // Convert Spark wrapped stages into PFA models
      val pfaStages = sparkStages.map { case (opStage, sparkStage, i) => toPFAModel(opStage, sparkStage) -> i }
      // Combine all stages and apply the original order
      val allStages = (opStages ++ pfaStages).sortBy(_._2).map(_._1)
      val resultFeatures = model.getResultFeatures().map(_.name).toSet

      // Score Function
      input: Map[String, Any] => {
        val inputMap = mutable.Map.empty ++= input
        val transformedRow = allStages.foldLeft(inputMap) {
          // For OP Models we simply call transform
          case (row, OPModel(output, stage)) =>
            row += output -> stage.transformKeyValue(row.apply)

          // For PFA Models we execute PFA engine action with json in/out
          case (row, PFAModel(inputs, (out, outCol), engine)) =>
            val inJson = rowToJson(row, inputs)
            val engineIn = engine.jsonInput(inJson)
            val engineOut = engine.action(engineIn)
            val resMap = parse(engineOut.toString).extract[Map[String, Any]]
            row += out -> resMap(outCol)
        }
        transformedRow.filterKeys(resultFeatures.contains).toMap
      }
    }

    /**
     * Convert Spark wrapped staged into PFA Models
     */
    private def toPFAModel(opStage: OPStage with SparkWrapperParams[_], sparkStage: Transformer): PFAModel = {
      // Update input/output params for Spark stages to default ones
      val inParam = sparkStage.getParam(inputCol.name)
      val outParam = sparkStage.getParam(outputCol.name)
      val inputs = opStage.getInputFeatures().map(_.name).map {
        case n if sparkStage.get(inParam).contains(n) => n -> inputCol.name
        case n if sparkStage.get(outParam).contains(n) => n -> outputCol.name
        case n => n -> n
      }.toMap
      val output = opStage.getOutputFeatureName
      sparkStage.set(inParam, inputCol.name).set(outParam, outputCol.name)
      val pfaJson = SparkSupport.toPFA(sparkStage, pretty = true)
      val pfaEngine = PFAEngine.fromJson(pfaJson).head
      PFAModel(inputs, (output, outputCol.name), pfaEngine)
    }

    /**
     * Convert row of Spark values into a json convertible Map
     * See [[FeatureTypeSparkConverter.toSpark]] for all possible values - we invert them here
     */
    private def rowToJson(row: mutable.Map[String, Any], inputs: Map[String, String]): String = {
      val in = inputs.map { case (k, v) => (v, row.get(k)) }.mapValues {
        case Some(v: Vector) => v.toArray
        case Some(v: mutable.WrappedArray[_]) => v.toArray(v.elemTag)
        case Some(v: Map[_, _]) => v.mapValues {
          case v: mutable.WrappedArray[_] => v.toArray(v.elemTag)
          case x => x
        }
        case None | Some(null) => null
        case Some(v) => v
      }
      Serialization.write(in)
    }
  }

}
