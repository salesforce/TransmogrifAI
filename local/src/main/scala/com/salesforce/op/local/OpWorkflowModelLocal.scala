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
import com.salesforce.op.features.types.{FeatureType, OPVector}
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import com.salesforce.op.stages.{OPStage, OpPipelineStage, OpTransformer}
import com.salesforce.op.utils.json.JsonUtils
import org.apache.spark.ml.{SparkMLSharedParamConstants, Transformer}
import org.apache.spark.ml.SparkMLSharedParamConstants._
import org.apache.spark.ml.linalg.Vector
import org.json4s._
import org.json4s.jackson.Serialization._

import scala.collection.mutable
import scala.collection.mutable.IndexedSeq

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

    implicit val formats: Formats = DefaultFormats

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
        inParam = sparkStage.getParam(inputCol.name)
        outParam = sparkStage.getParam(outputCol.name)
        inputs = s.getInputFeatures().map(_.name).map {
          case n if sparkStage.get(inParam).contains(n) => n -> inputCol.name
          case n if sparkStage.get(outParam).contains(n) => n -> outputCol.name
          case n => n -> n
        }.toMap
        output = s.getOutputFeatureName
        _ = sparkStage.set(inParam, inputCol.name).set(outParam, outputCol.name)
        pfaJson = SparkSupport.toPFA(sparkStage, pretty = true)
        pfaEngine = PFAEngine.fromJson(pfaJson).head
      } yield {
        println(sparkStage.extractParamMap())
        ((inputs, (output, outputCol.name), pfaEngine), i)
      }

      val allStages = (opStages ++ pfaEngines).sortBy(_._2)

      row => {
        val rowMap = collection.mutable.Map.empty ++ row
        val transformedRow = allStages.foldLeft(rowMap) {
          case (r, (s: OPStage with OpTransformer, i)) =>
            r += s.getOutputFeatureName -> s.transformKeyValue(r.apply)

          case (r, ((inputs: Map[String, String],
          output: (String, String),
          engine: PFAEngine[AnyRef, AnyRef]@unchecked), i)) =>
            val json = toPFAJson(r, inputs)
            println("INPUT: " + json)
            val engineIn = engine.jsonInput(json)
            val result = engine.action(engineIn)
            println("RESULT: " + result)
            val resMap = JsonUtils.fromString[Map[String, Any]](result.toString).get
            val (out, outCol) = output
            r += out -> resMap(outCol)
        }
        transformedRow.filterKeys(resultFeatures.contains).toMap
      }
    }

    private def toPFAJson(r: collection.mutable.Map[String, Any], inputs: Map[String, String]): String = {
      // Convert Spark values into a json convertible Map
      // See [[FeatureTypeSparkConverter.toSpark]] for all possible values
      val in = inputs.map { case (k, v) => (v, r.get(k)) }.mapValues {
        case None => null
        case Some(v: Vector) => v.toArray
        case Some(v: mutable.WrappedArray[_]) => v.toList
        case Some(v: Map[String, _]) => v.mapValues {
          case v: mutable.WrappedArray[_] => v.toList
          case x => x
        }
        case Some(v) => v
      }
      JsonUtils.toJsonString(in)
    }

  }

}
