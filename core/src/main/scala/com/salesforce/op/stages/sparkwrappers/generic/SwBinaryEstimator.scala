/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.OpPipelineStage2
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag


/**
 * Generic wrapper for any spark estimator that has two inputs and one output
 *
 * @param inputParam1Name name of spark parameter that sets the first input column
 * @param inputParam2Name name of spark parameter that sets the second input column
 * @param outputParamName name of spark parameter that sets the first output column
 * @param operationName   unique name of the operation this stage performs
 * @param sparkMlStageIn  spark estimator to wrap
 * @param uid             stage uid
 * @param tti1            type tag for first input
 * @param tti2            type tag for second input
 * @param tto             type tag for output
 * @param ttov            type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam O  output feature type
 * @tparam M  spark model type returned by spark estimator wrapped
 * @tparam E  spark estimator to wrap
 */
class SwBinaryEstimator[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType,
M <: Model[M], E <: Estimator[M] with Params]
(
  val inputParam1Name: String,
  val inputParam2Name: String,
  val outputParamName: String,
  val operationName: String,
  private val sparkMlStageIn: Option[E],
  val uid: String = UID[SwBinaryEstimator[I1, I2, O, M, E]]
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Estimator[SwBinaryModel[I1, I2, O, M]] with OpPipelineStage2[I1, I2, O] with SparkWrapperParams[E] {

  setSparkMlStage(sparkMlStageIn)
  set(sparkInputColParamNames, Array(inputParam1Name, inputParam2Name))
  set(sparkOutputColParamNames, Array(outputParamName))

  override def fit(dataset: Dataset[_]): SwBinaryModel[I1, I2, O, M] = {
    val model = getSparkMlStage().map { e =>
      val p1 = e.getParam(inputParam1Name)
      val p2 = e.getParam(inputParam2Name)
      val po = e.getParam(outputParamName)
      e.set(p1, in1.name).set(p2, in2.name).set(po, getOutputFeatureName).fit(dataset)
    }

    new SwBinaryModel[I1, I2, O, M](inputParam1Name, inputParam2Name, outputParamName, operationName, model, uid)
      .setParent(this)
      .setInput(in1.asFeatureLike[I1], in2.asFeatureLike[I2])
      .setOutputFeatureName(getOutputFeatureName)
  }

}


/**
 * Generic wrapper for any the model output by spark estimator that has two inputs and one output
 *
 * @param inputParam1Name name of spark parameter that sets the first input column
 * @param inputParam2Name name of spark parameter that sets the second input column
 * @param outputParamName name of spark parameter that sets the first output column
 * @param operationName   unique name of the operation this stage performs
 * @param sparkMlStageIn  spark estimator to wrap
 * @param uid             stage uid
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam O  type of output feature
 * @tparam T  type of spark model to wrap
 */
private[stages] final class SwBinaryModel[I1 <: FeatureType,
I2 <: FeatureType, O <: FeatureType, T <: Model[T] with Params]
(
  val inputParam1Name: String,
  val inputParam2Name: String,
  val outputParamName: String,
  val operationName: String,
  private val sparkMlStageIn: Option[T],
  val uid: String
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[SwBinaryModel[I1, I2, O, T]] with SwTransformer2[I1, I2, O, T] {

  setSparkMlStage(sparkMlStageIn)
}
