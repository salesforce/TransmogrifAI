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

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.OpPipelineStage4
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.Params
import org.apache.spark.sql._

import scala.reflect.runtime.universe.TypeTag

/**
 * Base class for wrapping spark transformers with four inputs and one output
 *
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam I4 fourth input feature type
 * @tparam O  output feature type
 * @tparam T  type of spark transformer to wrap
 */
private[stages] trait SwTransformer4[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType,
I4 <: FeatureType, O <: FeatureType, T <: Transformer with Params]
  extends Transformer with OpPipelineStage4[I1, I2, I3, I4, O] with SparkWrapperParams[T] {

  implicit def tti1: TypeTag[I1]
  implicit def tti2: TypeTag[I2]
  implicit def tti3: TypeTag[I3]
  implicit def tti4: TypeTag[I4]

  def inputParam1Name: String
  def inputParam2Name: String
  def inputParam3Name: String
  def inputParam4Name: String
  set(sparkInputColParamNames, Array(inputParam1Name, inputParam2Name, inputParam3Name, inputParam4Name))

  def outputParamName: String
  set(sparkOutputColParamNames, Array(outputParamName))

  override def transform(dataset: Dataset[_]): DataFrame = {
    getSparkMlStage().map { t =>
      val p1 = t.getParam(inputParam1Name)
      val p2 = t.getParam(inputParam2Name)
      val p3 = t.getParam(inputParam3Name)
      val p4 = t.getParam(inputParam4Name)
      val po = t.getParam(outputParamName)
      t.set(p1, in1.name).set(p2, in2.name).set(p3, in3.name).set(p4, in4.name).set(po, getOutputFeatureName)
        .transform(dataset)
    }.getOrElse(dataset.toDF())
  }

}

/**
 * Generic wrapper for any spark transformer that has four inputs and one output
 *
 * @param inputParam1Name name of spark parameter that sets the first input column
 * @param inputParam2Name name of spark parameter that sets the second input column
 * @param inputParam3Name name of spark parameter that sets the third input column
 * @param inputParam4Name name of spark parameter that sets the fourth input column
 * @param outputParamName name of spark parameter that sets the first output column
 * @param operationName   unique name of the operation this stage performs
 * @param sparkMlStageIn  spark estimator to wrap
 * @param uid             stage uid
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam I4 fourth input feature type
 * @tparam O  type of output feature
 * @tparam T  type of spark model to wrap
 */
class SwQuaternaryTransformer[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, I4 <: FeatureType,
O <: FeatureType, T <: Transformer with Params]
(
  val inputParam1Name: String,
  val inputParam2Name: String,
  val inputParam3Name: String,
  val inputParam4Name: String,
  val outputParamName: String,
  val operationName: String,
  private val sparkMlStageIn: Option[T],
  val uid: String = UID[SwQuaternaryTransformer[I1, I2, I3, I4, O, T]]
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tti3: TypeTag[I3],
  val tti4: TypeTag[I4],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends SwTransformer4[I1, I2, I3, I4, O, T] {

  setSparkMlStage(sparkMlStageIn)

}

