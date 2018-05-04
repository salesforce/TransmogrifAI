/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.OpPipelineStage4
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Generic wrapper for any spark estimator that has four inputs and one output
 *
 * @param inputParam1Name name of spark parameter that sets the first input column
 * @param inputParam2Name name of spark parameter that sets the second input column
 * @param inputParam3Name name of spark parameter that sets the third input column
 * @param inputParam4Name name of spark parameter that sets the fourth input column
 * @param outputParamName name of spark parameter that sets the first output column
 * @param operationName   unique name of the operation this stage performs
 * @param sparkMlStageIn  spark estimator to wrap
 * @param uid             stage uid
 * @param tti1            type tag for first input
 * @param tti2            type tag for second input
 * @param tti3            type tag for third input
 * @param tti4            type tag for fourth input
 * @param tto             type tag for output
 * @param ttov            type tag for output value
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam I3 third input feature type
 * @tparam I4 fourth input feature type
 * @tparam O  output feature type
 * @tparam M  spark model type returned by spark estimator wrapped
 * @tparam E  spark estimator to wrap
 */
class SwQuaternaryEstimator[I1 <: FeatureType, I2 <: FeatureType, I3 <: FeatureType, I4 <: FeatureType,
O <: FeatureType, M <: Model[M], E <: Estimator[M] with Params]
(
  val inputParam1Name: String,
  val inputParam2Name: String,
  val inputParam3Name: String,
  val inputParam4Name: String,
  val outputParamName: String,
  val operationName: String,
  private val sparkMlStageIn: Option[E],
  val uid: String = UID[SwQuaternaryEstimator[I1, I2, I3, I4, O, M, E]]
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tti3: TypeTag[I3],
  val tti4: TypeTag[I4],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Estimator[SwQuaternaryModel[I1, I2, I3, I4, O, M]]
  with OpPipelineStage4[I1, I2, I3, I4, O] with SparkWrapperParams[E] {

  setSparkMlStage(sparkMlStageIn)
  set(sparkInputColParamNames, Array(inputParam1Name, inputParam2Name, inputParam3Name, inputParam4Name))
  set(sparkOutputColParamNames, Array(outputParamName))

  override def fit(dataset: Dataset[_]): SwQuaternaryModel[I1, I2, I3, I4, O, M] = {
    val model = getSparkMlStage().map{ e =>
      val p1 = e.getParam(inputParam1Name)
      val p2 = e.getParam(inputParam2Name)
      val p3 = e.getParam(inputParam3Name)
      val p4 = e.getParam(inputParam4Name)
      val po = e.getParam(outputParamName)
      e.set(p1, in1.name).set(p2, in2.name).set(p3, in3.name).set(p4, in4.name).set(po, getOutputFeatureName)
        .fit(dataset)
    }

    new SwQuaternaryModel[I1, I2, I3, I4, O, M] (
      inputParam1Name,
      inputParam2Name,
      inputParam3Name,
      inputParam4Name,
      outputParamName,
      getOutputFeatureName,
      model,
      uid
    ).setParent(this).setInput(
      in1.asFeatureLike[I1],
      in2.asFeatureLike[I2],
      in3.asFeatureLike[I3],
      in4.asFeatureLike[I4]
    )
      .setInput(in1.asFeatureLike[I1], in2.asFeatureLike[I2], in3.asFeatureLike[I3], in4.asFeatureLike[I4])
      .setOutputFeatureName(getOutputFeatureName)
  }

}

/**
 * Generic wrapper for any the model output by spark estimator that has four inputs and one output
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
 * @tparam O  output feature type
 * @tparam T  type of spark transformer to wrap
 */
private[stages] final class SwQuaternaryModel[I1 <: FeatureType,
I2 <: FeatureType, I3 <: FeatureType, I4 <: FeatureType, O <: FeatureType, T <: Model[T] with Params]
(
  val inputParam1Name: String,
  val inputParam2Name: String,
  val inputParam3Name: String,
  val inputParam4Name: String,
  val outputParamName: String,
  val operationName: String,
  private val sparkMlStageIn: Option[T],
  val uid: String
) (
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tti3: TypeTag[I3],
  val tti4: TypeTag[I4],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends Model[SwQuaternaryModel[I1, I2, I3, I4, O, T]] with SwTransformer4[I1, I2, I3, I4, O, T] {

  setSparkMlStage(sparkMlStageIn)

}
