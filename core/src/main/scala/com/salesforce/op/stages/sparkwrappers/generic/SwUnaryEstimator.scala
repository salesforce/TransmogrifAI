/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.OpPipelineStage1
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Generic wrapper for any spark estimator that has one input and one output
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
class SwUnaryEstimator[I <: FeatureType, O <: FeatureType, M <: Model[M], E <: Estimator[M] with Params]
(
  val inputParamName: String,
  val outputParamName: String,
  val operationName: String,
  private val sparkMlStageIn: Option[E],
  val uid: String = UID[SwUnaryEstimator[I, O, M, E]]
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Estimator[SwUnaryModel[I, O, M]] with OpPipelineStage1[I, O] with SparkWrapperParams[E] {

  setSparkMlStage(sparkMlStageIn)
  set(sparkInputColParamNames, Array(inputParamName))
  set(sparkOutputColParamNames, Array(outputParamName))

  override def fit(dataset: Dataset[_]): SwUnaryModel[I, O, M] = {
    val model = getSparkMlStage().map { e =>
      val pi = e.getParam(inputParamName)
      val po = e.getParam(outputParamName)
      e.set(pi, in1.name).set(po, getOutputFeatureName).fit(dataset)
    }

    new SwUnaryModel[I, O, M](inputParamName, outputParamName, operationName, model, uid)
      .setParent(this)
      .setInput(in1.asFeatureLike[I])
      .setOutputFeatureName(getOutputFeatureName)
  }
}


/**
 * Generic wrapper for any the model output by spark estimator that has one input and one output
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
private[stages] final class SwUnaryModel[I <: FeatureType, O <: FeatureType, T <: Model[T] with Params]
(
  val inputParamName: String,
  val outputParamName: String,
  val operationName: String,
  private val sparkMlStageIn: Option[T],
  val uid: String
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[SwUnaryModel[I, O, T]] with SwTransformer1[I, O, T] with SparkWrapperParams[T] {

  setSparkMlStage(sparkMlStageIn)

}



