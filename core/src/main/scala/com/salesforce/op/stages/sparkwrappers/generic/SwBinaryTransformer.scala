/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.OpPipelineStage2
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.Params
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.reflect.runtime.universe.TypeTag

/**
 * Base class for wrapping spark transformers with two inputs and one output
 *
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 * @tparam O  output feature type
 * @tparam T  type of spark transformer to wrap
 */
private[stages] trait SwTransformer2[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType,
T <: Transformer with Params]
  extends Transformer with OpPipelineStage2[I1, I2, O] with SparkWrapperParams[T] {

  implicit def tti1: TypeTag[I1]
  implicit def tti2: TypeTag[I2]

  def inputParam1Name: String
  def inputParam2Name: String
  set(sparkInputColParamNames, Array(inputParam1Name, inputParam2Name))

  def outputParamName: String
  set(sparkOutputColParamNames, Array(outputParamName))

  override def transform(dataset: Dataset[_]): DataFrame = {
    getSparkMlStage().map { t =>
      val p1 = t.getParam(inputParam1Name)
      val p2 = t.getParam(inputParam2Name)
      val po = t.getParam(outputParamName)
      t.set(p1, in1.name).set(p2, in2.name).set(po, getOutputFeatureName).transform(dataset)
    }.getOrElse(dataset.toDF())
  }

}

/**
 * Generic wrapper for any spark transformer that has two inputs and one output
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
class SwBinaryTransformer[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType, T <: Transformer with Params]
(
  val inputParam1Name: String,
  val inputParam2Name: String,
  val outputParamName: String,
  val operationName: String,
  private val sparkMlStageIn: Option[T],
  val uid: String = UID[SwBinaryTransformer[I1, I2, O, T]]
)(
  implicit val tti1: TypeTag[I1],
  val tti2: TypeTag[I2],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends SwTransformer2[I1, I2, O, T] {

  setSparkMlStage(sparkMlStageIn)

}

