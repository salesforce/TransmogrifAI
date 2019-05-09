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

package com.salesforce.op.test

import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.OpPipelineStage
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.Params

import scala.reflect.ClassTag
import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Base test class for testing Spark transformer wrapper instances,
 * e.g [[SwUnaryTransformer]], [[SwBinaryTransformer]] etc.
 * Includes common tests for schema and data transformations.
 *
 * @tparam O                    output feature type
 * @tparam SparkTransformerType type of Spark transformer
 * @tparam TransformerType      type of Spark transformer wrapper being tested,
 *                              e.g. [[SwUnaryTransformer]], [[SwBinaryTransformer]] etc.
 */
abstract class SwTransformerSpec[O <: FeatureType,
SparkTransformerType <: Transformer with Params,
TransformerType <: OpPipelineStage[O] with Transformer with Params with SparkWrapperParams[SparkTransformerType]]
(
  implicit val cto: ClassTag[O], val wto: WeakTypeTag[O],
  val stc: ClassTag[SparkTransformerType], val ttc: ClassTag[TransformerType]
) extends OpPipelineStageSpec[O, TransformerType] with TransformerSpecCommon[O, TransformerType] {

  /**
   * The wrapped Spark stage instance
   */
  def sparkStage: Option[SparkTransformerType] = transformer.getSparkMlStage()

  it should "have a Spark stage set" in {
    sparkStage match {
      case None => fail("Spark stage is not set")
      case Some(s) =>
        withClue(s"Spark stage type is '${s.getClass.getName}' (expected '${stc.runtimeClass.getName}'):") {
          s.isInstanceOf[SparkTransformerType] shouldBe true
        }
    }
  }
  it should "have input column names set" in {
    transformer.getInputColParamNames() should not be empty
  }
  it should "have output column name set" in {
    transformer.getOutputColParamNames() should not be empty
  }
  it should "have inputs set on Spark stage" in {
    transformer.getInputColParamNames().flatMap(name => sparkStage.flatMap(s => s.get(s.getParam(name)))) shouldBe
      transformer.getInputFeatures().map(_.name)
  }
  it should "have output set on Spark stage" in {
    transformer.getOutputColParamNames().flatMap(name => sparkStage.flatMap(s => s.get(s.getParam(name)))) shouldBe
      Array(transformer.getOutputFeatureName)
  }

}
