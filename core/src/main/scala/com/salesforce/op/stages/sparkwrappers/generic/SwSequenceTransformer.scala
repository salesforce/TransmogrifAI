/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.UID
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.OpPipelineStageN
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.Params
import org.apache.spark.sql._

import scala.reflect.runtime.universe.TypeTag

/**
 * Base class for wrapping spark transformers with has a sequence of inputs of the same type and one output
 *
 * @tparam I input feature type
 * @tparam O output feature type
 * @tparam T type of spark transformer to wrap
 */
private[stages] trait SwTransformerN[I <: FeatureType, O <: FeatureType, T <: Transformer with Params]
  extends Transformer with OpPipelineStageN[I, O] with SparkWrapperParams[T] {

  implicit def tti: TypeTag[I]

  def inputParamName: String
  set(sparkInputColParamNames, Array(inputParamName))

  def outputParamName: String
  set(sparkOutputColParamNames, Array(outputParamName))

  override def transform(dataset: Dataset[_]): DataFrame = {
    getSparkMlStage().map { t =>
      val p = t.getParam(inputParamName)
      val po = t.getParam(outputParamName)
      t.set(p, inN.map(_.name)).set(po, getOutputFeatureName).transform(dataset)
    }.getOrElse(dataset.toDF())
  }

}

/**
 * Generic wrapper for any spark transformer that has a sequence of inputs of the same type and one output
 *
 * @param inputParamName  name of spark parameter that sets the first input column
 * @param outputParamName name of spark parameter that sets the first output column
 * @param operationName   unique name of the operation this stage performs
 * @param sparkMlStageIn  spark estimator to wrap
 * @param uid             stage uid
 * @tparam I input feature type
 * @tparam O output feature type
 * @tparam T type of spark transformer to wrap
 */
class SwSequenceTransformer[I <: FeatureType, O <: FeatureType, T <: Transformer with Params]
(
  val inputParamName: String,
  val outputParamName: String,
  val operationName: String,
  private val sparkMlStageIn: Option[T],
  val uid: String = UID[SwSequenceTransformer[I, O, T]]
)(
  implicit val tti: TypeTag[I],
  val tto: TypeTag[O],
  val ttov: TypeTag[O#Value]
) extends SwTransformerN[I, O, T] {

  setSparkMlStage(sparkMlStageIn)

}

