/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.stages.{OpPipelineStage2to3, _}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql._

import scala.reflect.runtime.universe.TypeTag

/**
 * Generic wrapper for any spark estimator which has two inputs and three outputs
 *
 * @param inputParam1Name     name of spark parameter that sets the first input column
 * @param inputParam2Name     name of spark parameter that sets the second input column
 * @param outputParam1Name    name of spark parameter that sets the first output column
 * @param outputParam2Name    name of spark parameter that sets the second output column
 * @param outputParam3Name    name of spark parameter that sets the third output column
 * @param stage1OperationName unique name of the operation first stage performs
 * @param stage2OperationName unique name of the operation second stage performs
 * @param stage3OperationName unique name of the operation third stage performs
 * @param sparkMlStageIn      instance of spark estimator to wrap
 * @param uid                 stage uid
 * @param i1ttag              type tag for first input
 * @param i2ttag              type tag for second input
 * @param o1ttag              type tag for first output
 * @param o2ttag              type tag for second output
 * @param o3ttag              type tag for third output
 * @param i1ttiv              type tag for first input value
 * @param i2ttiv              type tag for second input value
 * @param o1ttov              type tag for first output value
 * @param o2ttov              type tag for second output value
 * @param o3ttov              type tag for third output value
 * @tparam I1 input feature type 1
 * @tparam I2 input feature type 2
 * @tparam O1 first output feature type
 * @tparam O2 second output feature type
 * @tparam O3 third output feature type
 * @tparam M  spark model type returned by spark estimator wrapped
 * @tparam E  spark estimator to wrap
 */
class SwThreeStageBinaryEstimator[I1 <: FeatureType, I2 <: FeatureType, O1 <: FeatureType, O2 <: FeatureType,
O3 <: FeatureType, M <: Model[M], E <: Estimator[M]]
(
  val inputParam1Name: String,
  val inputParam2Name: String,
  val outputParam1Name: String,
  val outputParam2Name: String,
  val outputParam3Name: String,
  val stage1OperationName: String,
  val stage2OperationName: String,
  val stage3OperationName: String,
  private val sparkMlStageIn: Option[E],
  val uid: String = UID[SwThreeStageBinaryEstimator[I1, I2, O1, O2, O3, M, E]]
)(
  implicit val i1ttag: TypeTag[I1],
  val i2ttag: TypeTag[I2],
  val o1ttag: TypeTag[O1],
  val o2ttag: TypeTag[O2],
  val o3ttag: TypeTag[O3],
  val i1ttiv: TypeTag[I1#Value],
  val i2ttiv: TypeTag[I2#Value],
  val o1ttov: TypeTag[O1#Value],
  val o2ttov: TypeTag[O2#Value],
  val o3ttov: TypeTag[O3#Value]
) extends Estimator[SwThreeStageBinaryModel[I1, I2, O1, O2, O3, M]]
  with OpPipelineStage2to3[I1, I2, O1, O2, O3] with SparkWrapperParams[E] {

  setSparkMlStage(sparkMlStageIn)
  set(sparkInputColParamNames, Array(inputParam1Name, inputParam2Name))
  set(sparkOutputColParamNames, Array(outputParam1Name, outputParam2Name, outputParam3Name))

  private lazy val stage1uid = UID[SwBinaryEstimator[I1, I2, O1, M, E]]
  private lazy val stage2uid = UID[SwTernaryTransformer[I1, I2, O1, O2, M]]
  private lazy val stage3uid = UID[SwQuaternaryTransformer[I1, I2, O1, O2, O3, M]]

  private lazy val outputName1 = makeOutputNameFromStageId[O1](stage1uid, Seq(in1, in2))
  private lazy val outputName2 = makeOutputNameFromStageId[O2](stage2uid, Seq(in1, in2), 2)
  private lazy val outputName3 = makeOutputNameFromStageId[O3](stage3uid, Seq(in1, in2), 3)

  // put together parameter names and values
  private lazy val outputs = $(sparkOutputColParamNames).zip(
    Array(outputName1, outputName2, outputName3))

  private[op] lazy val stage1 = new SwBinaryEstimatorSpecial[I1, I2, O1, M, E](
    inputParam1Name = $(sparkInputColParamNames)(0),
    inputParam2Name = $(sparkInputColParamNames)(1),
    outputParamName = $(sparkOutputColParamNames)(0),
    operationName = stage1OperationName,
    sparkMlStageIn = getSparkMlStage().map { spk => // set all the outputs for this stage
      outputs.foldLeft(spk) { case (s, (pname, pvalue)) => s.set(s.getParam(pname), pvalue) }
    },
    uid = stage1uid,
    outputs
  ).setInput(in1.asFeatureLike[I1], in2.asFeatureLike[I2])

  private[op] lazy val stage2 = new SwTernaryTransformer[I1, I2, O1, O2, M](
    inputParam1Name = $(sparkInputColParamNames)(0),
    inputParam2Name = $(sparkInputColParamNames)(1),
    inputParam3Name = stage1OperationName,
    outputParamName = $(sparkOutputColParamNames)(1),
    operationName = stage2OperationName,
    sparkMlStageIn = None,
    uid = stage2uid
  ).setInput(in1.asFeatureLike[I1], in2.asFeatureLike[I2], stage1.getOutput())

  private[op] lazy val stage3 = new SwQuaternaryTransformer[I1, I2, O1, O2, O3, M](
    inputParam1Name = $(sparkInputColParamNames)(0),
    inputParam2Name = $(sparkInputColParamNames)(1),
    inputParam3Name = stage1OperationName,
    inputParam4Name = stage2OperationName,
    outputParamName = $(sparkOutputColParamNames)(2),
    operationName = stage3OperationName,
    sparkMlStageIn = None,
    uid = stage3uid
  ).setInput(in1.asFeatureLike[I1], in2.asFeatureLike[I2], stage1.getOutput(), stage2.getOutput())

  /**
   * Output features that will be created by the transformation
   *
   * @return features of type O1, O2 and O3
   */
  final override def getOutput(): (FeatureLike[O1], FeatureLike[O2], FeatureLike[O3]) = {
    (stage1.getOutput(), stage2.getOutput(), stage3.getOutput())
  }

  override def fit(dataset: Dataset[_]): SwThreeStageBinaryModel[I1, I2, O1, O2, O3, M] = {
    val model = stage1.fit(dataset)

    new SwThreeStageBinaryModel[I1, I2, O1, O2, O3, M](
      inputParam1Name,
      inputParam2Name,
      outputParam1Name,
      outputParam2Name,
      outputParam3Name,
      stage1OperationName,
      stage2OperationName,
      stage3OperationName,
      model,
      stage2,
      stage3,
      uid
    ).setParent(this).setInput(in1.asFeatureLike[I1], in2.asFeatureLike[I2])

  }
}

/**
 * Generic wrapper for any model returned by an estimator which has two inputs and three outputs
 *
 * @param inputParam1Name     name of spark parameter that sets the first input column
 * @param inputParam2Name     name of spark parameter that sets the second input column
 * @param outputParam1Name    name of spark parameter that sets the first output column
 * @param outputParam2Name    name of spark parameter that sets the second output column
 * @param outputParam3Name    name of spark parameter that sets the third output column
 * @param stage1OperationName unique name of the operation first stage performs
 * @param stage2OperationName unique name of the operation second stage performs
 * @param stage3OperationName unique name of the operation third stage performs
 * @param stage1              first wrapping stage for output one (this is the only stage that actually does anything)
 * @param stage2              second stage - dummy for generating second output
 * @param stage3              third stage - dummy for generating third output
 * @param uid                 stage uid
 * @tparam I1 input feature type 1
 * @tparam I2 input feature type 2
 * @tparam O1 first output feature type
 * @tparam O2 second output feature type
 * @tparam O3 third output feature type
 * @tparam M
 */
private[stages] final class SwThreeStageBinaryModel[I1 <: FeatureType, I2 <: FeatureType, O1 <: FeatureType,
O2 <: FeatureType, O3 <: FeatureType, M <: Model[M]]
(
  val inputParam1Name: String,
  val inputParam2Name: String,
  val outputParam1Name: String,
  val outputParam2Name: String,
  val outputParam3Name: String,
  val stage1OperationName: String,
  val stage2OperationName: String,
  val stage3OperationName: String,
  val stage1: SwBinaryModel[I1, I2, O1, M],
  val stage2: SwTernaryTransformer[I1, I2, O1, O2, M],
  val stage3: SwQuaternaryTransformer[I1, I2, O1, O2, O3, M],
  val uid: String
) extends Model[SwThreeStageBinaryModel[I1, I2, O1, O2, O3, M]]
  with OpPipelineStage2to3[I1, I2, O1, O2, O3] with SparkWrapperParams[M] {

  setSparkMlStage(stage1.getSparkMlStage())
  set(sparkInputColParamNames, Array(inputParam1Name, inputParam2Name))
  set(sparkOutputColParamNames, Array(outputParam1Name, outputParam2Name, outputParam3Name))

  override def transform(dataset: Dataset[_]): DataFrame = stage1.transform(dataset)

  override def getOutput(): (FeatureLike[O1], FeatureLike[O2], FeatureLike[O3]) =
    (stage1.getOutput(), stage2.getOutput(), stage3.getOutput())
}

/**
 * Wrapper for any spark estimator that has two inputs and three outputs (for use in three stage wrapper)
 */
private[op] class SwBinaryEstimatorSpecial[I1 <: FeatureType, I2 <: FeatureType, O <: FeatureType,
M <: Model[M], E <: Estimator[M]]
(
  inputParam1Name: String,
  inputParam2Name: String,
  outputParamName: String,
  operationName: String,
  private val sparkMlStageIn: Option[E],
  uid: String = UID[SwBinaryEstimator[I1, I2, O, M, E]],
  val outputNames: Array[(String, String)]
)(
  implicit tti1: TypeTag[I1],
  tti2: TypeTag[I2],
  tto: TypeTag[O],
  ttov: TypeTag[O#Value]
) extends SwBinaryEstimator[I1, I2, O, M, E] (inputParam1Name = inputParam1Name, inputParam2Name = inputParam2Name,
  outputParamName = outputParamName, operationName = operationName, sparkMlStageIn = sparkMlStageIn,
  uid = uid)(tti1 = tti1, tti2 = tti2, tto = tto, ttov = ttov){

  override def setOutputFeatureName(m: String): this.type = {
    getSparkMlStage().map { spk => // set all the outputs for this stage
      outputNames.zipWithIndex.foldLeft(spk) { case (s, ((pname, pvalue), i)) =>
        val newName = updateOutputName(m, pvalue, i)
        s.set(s.getParam(pname), newName)
      }}
    set(outputFeatureName, m)
  }
}
