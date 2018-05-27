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
import com.salesforce.op.stages.OpPipelineStageN
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Generic wrapper for any spark estimator that has a sequence of inputs of the same type and one output
 *
 * @param inputParamName  name of spark parameter that sets the second input column
 * @param outputParamName name of spark parameter that sets the first output column
 * @param operationName   unique name of the operation this stage performs
 * @param sparkMlStageIn  spark estimator to wrap
 * @param uid             stage uid
 * @param tti             type tag for input
 * @param tto             type tag for output
 * @param ttov            type tag for output value
 * @tparam I input feature type
 * @tparam O output feature type
 * @tparam M spark model type returned by spark estimator wrapped
 * @tparam E spark estimator to wrap
 */
class SwSequenceEstimator[I <: FeatureType, O <: FeatureType, M <: Model[M], E <: Estimator[M] with Params]
(
  val inputParamName: String,
  val operationName: String,
  val outputParamName: String,
  private val sparkMlStageIn: Option[E],
  val uid: String = UID[SwSequenceEstimator[I, O, M, E]]
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Estimator[SwSequenceModel[I, O, M]] with OpPipelineStageN[I, O] with SparkWrapperParams[E] {

  setSparkMlStage(sparkMlStageIn)
  set(sparkInputColParamNames, Array(inputParamName))
  set(sparkOutputColParamNames, Array(outputParamName))

  override def fit(dataset: Dataset[_]): SwSequenceModel[I, O, M] = {
    val model = getSparkMlStage().map{ e =>
      val pi = e.getParam(inputParamName)
      val po = e.getParam(outputParamName)
      e.set(pi, inN.map(_.name)).set(po, getOutputFeatureName).fit(dataset)
    }

    new SwSequenceModel[I, O, M](inputParamName, outputParamName, operationName, model, uid)
      .setParent(this)
      .setInput(inN.map(_.asFeatureLike[I]))
      .setOutputFeatureName(getOutputFeatureName)
  }
}

/**
 * Generic wrapper for any the model output by spark estimator that has a sequence of inputs of the same type
 * and one output
 *
 * @param inputParamName  name of spark parameter that sets the first input column
 * @param outputParamName name of spark parameter that sets the second input column
 * @param operationName   unique name of the operation this stage performs
 * @param sparkMlStageIn  spark estimator to wrap
 * @param uid             stage uid
 * @tparam I input feature type
 * @tparam O type of output feature
 * @tparam T type of spark model to wrap
 */
private[stages] final class SwSequenceModel[I <: FeatureType, O <: FeatureType, T <: Model[T] with Params]
(
  val inputParamName: String,
  val operationName: String,
  val outputParamName: String,
  private val sparkMlStageIn: Option[T],
  val uid: String
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[SwSequenceModel[I, O, T]] with SwTransformerN[I, O, T] {

  setSparkMlStage(sparkMlStageIn)

}
